#!/usr/bin/env python3
"""Detect incoherent pseudo-detail from 2D fragmentation and 3D Gaussian mixing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from data_processing.camera_trajectories import (
    applied_transform_matrix,
    camera_intrinsics_for_frame,
    opengl_c2w_to_opencv_w2c,
)
from worldact.data.gaussians import GaussianScene
from worldact.data.poses import CameraPose
from worldact.rendering.rasterizer import SH_C0, rasterize_gaussian_contributors


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def camera_from_frame(transforms: dict, frame: dict, device: torch.device) -> CameraPose:
    intrinsics = camera_intrinsics_for_frame(transforms, frame)
    w2c = opengl_c2w_to_opencv_w2c(frame["transform_matrix"], applied_transform_matrix(transforms))
    intrinsic = np.array(
        [
            [float(intrinsics["fl_x"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fl_y"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return CameraPose(
        extrinsic=torch.from_numpy(w2c.astype(np.float32)).to(device),
        intrinsic=torch.from_numpy(intrinsic).to(device),
        width=int(intrinsics["w"]),
        height=int(intrinsics["h"]),
    )


def image_fragmentation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    gradient_energy = cv2.boxFilter(gradient, -1, (9, 9))
    laplacian_energy = cv2.boxFilter(laplacian, -1, (9, 9))

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2) / (jxx + jyy + 1e-6)

    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    weighted_bins = []
    for index in range(8):
        selected = ((angle >= index * np.pi / 8) & (angle < (index + 1) * np.pi / 8)).astype(np.float32)
        weighted_bins.append(cv2.boxFilter(selected * gradient, -1, (9, 9), normalize=False))
    bins = np.maximum(np.stack(weighted_bins), 0.0)
    probability = bins / (bins.sum(axis=0, keepdims=True) + 1e-8)
    entropy = -(probability * np.log2(probability + 1e-8)).sum(axis=0) / 3.0
    high_frequency = np.sqrt(np.maximum(gradient_energy * laplacian_energy, 0.0))
    fragmentation = high_frequency * entropy * (1.0 - 0.55 * coherence)
    fragmentation = np.nan_to_num(fragmentation, nan=0.0, posinf=0.0, neginf=0.0)
    return fragmentation.astype(np.float32), entropy.astype(np.float32), coherence.astype(np.float32), high_frequency.astype(np.float32)


def robust_bounds(samples: list[np.ndarray], low: float = 0.10, high: float = 0.90) -> tuple[float, float]:
    values = np.concatenate(samples)
    lo, hi = np.quantile(values, (low, high))
    return float(lo), float(max(hi, lo + 1e-8))


def normalize(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    x = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def save_gray(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB").save(path)


def colorize(values: np.ndarray) -> np.ndarray:
    palette = np.array(
        ((0.06, 0.22, 0.72), (0.00, 0.82, 1.00), (1.00, 0.86, 0.00), (1.00, 0.16, 0.56)),
        dtype=np.float32,
    )
    bins = np.digitize(np.clip(values, 0.0, 1.0), (0.35, 0.55, 0.72))
    return palette[bins]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--roi-dir", type=Path, required=True)
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--missing-alpha-low", type=float, default=0.03)
    parser.add_argument("--missing-alpha-high", type=float, default=0.35)
    parser.add_argument("--disorder-low", type=float, default=0.42)
    parser.add_argument("--disorder-high", type=float, default=0.68)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepared = args.prepared_dir.resolve()
    roi_dir = args.roi_dir.resolve()
    direction_dir = args.direction_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.replace:
        raise FileExistsError(f"{output} is not empty; pass --replace")
    directories = (
        "raw_metrics", "texture_disorder", "texture_disorder_color", "diffusion_edit_priority", "region_overlay",
        "region_high_precision", "region_low_precision", "region_missing", "noise_strength_preview",
    )
    for directory in directories:
        (output / directory).mkdir(parents=True, exist_ok=True)

    split = json.loads((prepared / "split.json").read_text())
    metadata = next(iter(split["test"].values()))
    transforms = json.loads(resolve_path(prepared, metadata["transforms_path"]).read_text())
    frames = transforms["frames"]
    targets = [int(index) for index in json.loads(resolve_path(prepared, metadata["target_indices_path"]).read_text())]
    render_dir = resolve_path(prepared, metadata["render_dir"])
    scene = GaussianScene.from_ply(resolve_path(prepared, metadata["initial_object_ply"]))
    device = torch.device(args.device)
    scene.params = scene.params.to(device)
    means = scene.params.means
    dc_color = (scene.params.colors[:, :3] * SH_C0 + 0.5).clamp(0.0, 1.0)
    direction_disorder = torch.from_numpy(
        np.load(direction_dir / "gaussian_orientation_disorder.npy", allow_pickle=False).astype(np.float32)
    ).to(device)

    sample_values: dict[str, list[np.ndarray]] = {key: [] for key in ("fragmentation", "depth_std", "color_std", "direction")}
    frame_cache: dict[int, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(20260813)
    with torch.inference_mode():
        for index in targets:
            name = f"{index:05d}.png"
            camera = camera_from_frame(transforms, frames[index], device)
            contributors = rasterize_gaussian_contributors(scene, camera)
            gids = contributors["gaussian_ids"].long()
            pids = contributors["pixel_ids"].long()
            weights = contributors["weights"].float()
            count = camera.height * camera.width
            mass = torch.bincount(pids, weights=weights, minlength=count)

            means_camera = means @ camera.extrinsic[:3, :3].T + camera.extrinsic[:3, 3]
            depth = means_camera[:, 2]
            depth_1 = torch.bincount(pids, weights=weights * depth[gids], minlength=count) / mass.clamp_min(1e-8)
            depth_2 = torch.bincount(pids, weights=weights * depth[gids].square(), minlength=count) / mass.clamp_min(1e-8)
            depth_std = torch.sqrt((depth_2 - depth_1.square()).clamp_min(0.0))

            color_variance = torch.zeros(count, device=device)
            for channel in range(3):
                values = dc_color[:, channel]
                mean = torch.bincount(pids, weights=weights * values[gids], minlength=count) / mass.clamp_min(1e-8)
                second = torch.bincount(pids, weights=weights * values[gids].square(), minlength=count) / mass.clamp_min(1e-8)
                color_variance += (second - mean.square()).clamp_min(0.0)
            color_std = torch.sqrt(color_variance / 3.0)
            direction = torch.bincount(
                pids, weights=weights * direction_disorder[gids], minlength=count
            ) / mass.clamp_min(1e-8)

            opacity = mass.reshape(camera.height, camera.width).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            rendered = np.asarray(Image.open(render_dir / name).convert("RGB"), dtype=np.float32) / 255.0
            fragmentation, entropy, coherence, high_frequency = image_fragmentation(rendered)
            roi = np.load(roi_dir / f"{index:05d}.npy", allow_pickle=False).astype(np.float32)
            valid = (roi > 0.25) & (opacity > args.missing_alpha_high)
            raw = {
                "fragmentation": fragmentation,
                "depth_std": depth_std.reshape(camera.height, camera.width).cpu().numpy().astype(np.float32),
                "color_std": color_std.reshape(camera.height, camera.width).cpu().numpy().astype(np.float32),
                "direction": direction.reshape(camera.height, camera.width).cpu().numpy().astype(np.float32),
                "entropy": entropy,
                "coherence": coherence,
                "high_frequency": high_frequency,
                "opacity": opacity,
                "roi": roi,
                "rendered": rendered,
            }
            frame_cache[index] = raw
            for key in sample_values:
                values = raw[key][valid]
                if values.size > 20000:
                    values = rng.choice(values, 20000, replace=False)
                sample_values[key].append(values)
            np.savez_compressed(output / "raw_metrics" / f"{index:05d}.npz", **{k: v for k, v in raw.items() if k != "rendered"})

    bounds = {key: robust_bounds(values) for key, values in sample_values.items()}
    weights = {"fragmentation": 0.55, "depth_std": 0.20, "color_std": 0.15, "direction": 0.10}
    stats = {}
    for index, raw in frame_cache.items():
        name = f"{index:05d}.png"
        normalized = {key: normalize(raw[key], bounds[key]) for key in bounds}
        mixing = 0.45 * normalized["depth_std"] + 0.35 * normalized["color_std"] + 0.20 * normalized["direction"]
        disorder = sum(weights[key] * normalized[key] for key in weights)
        # A strong 2D response without any 3D mixing evidence is often a real edge or marking.
        disorder *= 0.65 + 0.35 * mixing
        disorder = cv2.GaussianBlur(disorder.astype(np.float32), (0, 0), 1.2)
        disorder = np.clip(disorder, 0.0, 1.0)

        missing_score = 1.0 - smoothstep(args.missing_alpha_low, args.missing_alpha_high, raw["opacity"])
        missing = raw["roi"] * missing_score
        nonmissing = raw["roi"] * (1.0 - missing_score)
        low_gate = smoothstep(args.disorder_low, args.disorder_high, disorder)
        low = nonmissing * low_gate
        high = nonmissing * (1.0 - low_gate)
        noise = np.clip(0.35 * low + 0.85 * missing, 0.0, 1.0)

        alpha = (raw["roi"] * 0.72)[..., None]
        overlay = raw["rendered"] * (1.0 - alpha) + colorize(disorder) * alpha
        region_color = (
            high[..., None] * np.array((0.12, 0.78, 0.35), np.float32)
            + low[..., None] * np.array((1.00, 0.48, 0.05), np.float32)
            + missing[..., None] * np.array((0.92, 0.10, 0.68), np.float32)
        )
        region_alpha = np.clip(high + low + missing, 0.0, 1.0)[..., None] * 0.65
        region_overlay = raw["rendered"] * (1.0 - region_alpha) + region_color * 0.65
        save_gray(output / "texture_disorder" / name, disorder * raw["roi"])
        np.save(output / "texture_disorder" / f"{index:05d}.npy", disorder)
        save_rgb(output / "texture_disorder_color" / name, overlay)
        save_rgb(output / "region_overlay" / name, region_overlay)
        save_gray(output / "diffusion_edit_priority" / name, noise)
        np.save(output / "diffusion_edit_priority" / f"{index:05d}.npy", noise.astype(np.float32))
        save_gray(output / "region_high_precision" / name, high)
        save_gray(output / "region_low_precision" / name, low)
        np.save(output / "region_low_precision" / f"{index:05d}.npy", low.astype(np.float32))
        save_gray(output / "region_missing" / name, missing)
        np.save(output / "region_missing" / f"{index:05d}.npy", missing.astype(np.float32))
        save_gray(output / "noise_strength_preview" / name, noise)
        total = max(float(raw["roi"].sum()), 1e-8)
        stats[str(index)] = {
            "mean_disorder_nonmissing": float((disorder * nonmissing).sum() / max(float(nonmissing.sum()), 1e-8)),
            "high_fraction": float(high.sum() / total),
            "low_fraction": float(low.sum() / total),
            "missing_fraction": float(missing.sum() / total),
        }

    summary = {
        "prepared_dir": str(prepared), "roi_dir": str(roi_dir), "direction_dir": str(direction_dir),
        "normalization_bounds_p10_p90": {key: list(value) for key, value in bounds.items()},
        "weights": weights,
        "classification_thresholds": {"disorder_low": args.disorder_low, "disorder_high": args.disorder_high},
        "frame_stats": stats,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote texture-disorder analysis to {output}")


if __name__ == "__main__":
    main()
