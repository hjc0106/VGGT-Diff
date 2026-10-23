#!/usr/bin/env python3
"""Visualize local 3D Gaussian orientation consistency on target camera views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from data_processing.camera_trajectories import (
    applied_transform_matrix,
    camera_intrinsics_for_frame,
    opengl_c2w_to_opencv_w2c,
)
from worldact.data.gaussians import GaussianScene
from worldact.data.poses import CameraPose
from worldact.rendering.rasterizer import rasterize_gaussian_contributors


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


def quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quaternions.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)


def gaussian_orientation_metrics(
    scene: GaussianScene,
    neighbors: int,
    minimum_planarity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = scene.params.means.detach().cpu().numpy().astype(np.float32)
    scales = np.exp(scene.params.scales.detach().cpu().numpy().astype(np.float32))
    rotations = scene.params.rotations.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(scales, axis=1)
    minimum_axis = order[:, 0]
    sorted_scales = np.take_along_axis(scales, order, axis=1)
    planarity = np.clip(1.0 - sorted_scales[:, 0] / np.maximum(sorted_scales[:, 1], 1e-8), 0.0, 1.0)

    matrices = quaternion_wxyz_to_matrix(rotations)
    normals = matrices[np.arange(len(matrices)), :, minimum_axis]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

    tree = cKDTree(means)
    distances, indices = tree.query(means, k=min(neighbors + 1, len(means)), workers=-1)
    distances = distances[:, 1:].astype(np.float32)
    indices = indices[:, 1:]
    local_radius = np.maximum(distances[:, -1:], 1e-8)
    distance_weight = np.exp(-2.0 * np.square(distances / local_radius))
    neighbor_planarity = planarity[indices]
    reliability_weight = distance_weight * neighbor_planarity
    dot = np.abs(np.einsum("ij,ikj->ik", normals, normals[indices]))
    consistency = np.sum(reliability_weight * dot, axis=1) / np.maximum(
        np.sum(reliability_weight, axis=1), 1e-8
    )
    reliable_neighbor_fraction = np.mean(neighbor_planarity >= minimum_planarity, axis=1)
    reliability = np.sqrt(planarity * np.clip(np.mean(neighbor_planarity, axis=1), 0.0, 1.0))
    reliability *= reliable_neighbor_fraction
    consistency = np.where(reliability > 1e-6, consistency, 0.0).astype(np.float32)
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32)
    disorder = ((1.0 - consistency) * reliability).astype(np.float32)
    return consistency, reliability, disorder


def save_gray(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path)


def colorize_consistency(values: np.ndarray) -> np.ndarray:
    """Use discrete, perceptually separated bands for visual diagnosis."""
    palette = np.array(
        (
            (0.30, 0.08, 0.72),  # < 0.55: violet
            (1.00, 0.10, 0.62),  # 0.55-0.70: magenta
            (0.00, 0.82, 1.00),  # 0.70-0.82: cyan
            (1.00, 0.88, 0.00),  # 0.82-0.90: yellow
            (0.18, 1.00, 0.28),  # >= 0.90: lime
        ),
        dtype=np.float32,
    )
    bins = np.digitize(np.clip(values, 0.0, 1.0), (0.55, 0.70, 0.82, 0.90))
    return palette[bins]


def colorize_disorder(values: np.ndarray) -> np.ndarray:
    low = np.array((0.12, 0.38, 0.85), dtype=np.float32)
    high = np.array((0.93, 0.13, 0.68), dtype=np.float32)
    return low + np.clip(values, 0.0, 1.0)[..., None] * (high - low)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--roi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--minimum-planarity", type=float, default=0.15)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepared = args.prepared_dir.resolve()
    roi_dir = args.roi_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.replace:
        raise FileExistsError(f"{output} is not empty; pass --replace")
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "orientation_consistency",
        "orientation_consistency_color",
        "orientation_reliability",
        "orientation_reliability_color",
        "orientation_disorder",
        "orientation_disorder_color",
    )
    for name in names:
        (output / name).mkdir(parents=True, exist_ok=True)

    split = json.loads((prepared / "split.json").read_text())
    metadata = next(iter(split["test"].values()))
    transforms = json.loads(resolve_path(prepared, metadata["transforms_path"]).read_text())
    frames = transforms["frames"]
    target_indices = json.loads(resolve_path(prepared, metadata["target_indices_path"]).read_text())
    render_dir = resolve_path(prepared, metadata["render_dir"])
    scene = GaussianScene.from_ply(resolve_path(prepared, metadata["initial_object_ply"]))
    consistency, reliability, disorder = gaussian_orientation_metrics(
        scene, args.neighbors, args.minimum_planarity
    )
    np.save(output / "gaussian_orientation_consistency.npy", consistency)
    np.save(output / "gaussian_orientation_reliability.npy", reliability)
    np.save(output / "gaussian_orientation_disorder.npy", disorder)

    device = torch.device(args.device)
    scene.params = scene.params.to(device)
    gaussian_values = {
        "consistency": torch.from_numpy(consistency).to(device),
        "reliability": torch.from_numpy(reliability).to(device),
        "disorder": torch.from_numpy(disorder).to(device),
    }
    frame_stats: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for index in target_indices:
            index = int(index)
            name = f"{index:05d}.png"
            camera = camera_from_frame(transforms, frames[index], device)
            contributors = rasterize_gaussian_contributors(scene, camera)
            gids = contributors["gaussian_ids"].long()
            pids = contributors["pixel_ids"].long()
            weights = contributors["weights"].float()
            count = camera.height * camera.width
            mass = torch.bincount(pids, weights=weights, minlength=count)
            maps = {}
            for key, values in gaussian_values.items():
                weighted = torch.bincount(pids, weights=weights * values[gids], minlength=count)
                maps[key] = (weighted / mass.clamp_min(1e-8)).reshape(camera.height, camera.width)
            opacity = mass.reshape(camera.height, camera.width).clamp(0.0, 1.0)
            support = (opacity > 0.01).float()
            maps = {key: value.cpu().numpy().astype(np.float32) for key, value in maps.items()}
            support_np = support.cpu().numpy().astype(np.float32)
            roi = np.load(roi_dir / f"{index:05d}.npy", allow_pickle=False).astype(np.float32)
            valid = roi * support_np
            rendered = np.asarray(Image.open(render_dir / name).convert("RGB"), dtype=np.float32) / 255.0

            consistency_color = colorize_consistency(maps["consistency"])
            disorder_color = colorize_disorder(maps["disorder"])
            reliability_color = np.repeat(maps["reliability"][..., None], 3, axis=2)
            alpha = (valid * 0.68)[..., None]
            overlays = {
                "orientation_consistency_color": rendered * (1.0 - alpha) + consistency_color * alpha,
                "orientation_reliability_color": rendered * (1.0 - alpha) + reliability_color * alpha,
                "orientation_disorder_color": rendered * (1.0 - alpha) + disorder_color * alpha,
            }
            for directory, values in overlays.items():
                save_rgb(output / directory / name, values)
            save_gray(output / "orientation_consistency" / name, maps["consistency"] * valid)
            save_gray(output / "orientation_reliability" / name, maps["reliability"] * valid)
            save_gray(output / "orientation_disorder" / name, maps["disorder"] * valid)
            np.save(output / "orientation_consistency" / f"{index:05d}.npy", maps["consistency"])

            valid_sum = max(float(valid.sum()), 1e-8)
            low = valid * (maps["consistency"] < 0.70)
            medium = valid * ((maps["consistency"] >= 0.70) & (maps["consistency"] < 0.90))
            high = valid * (maps["consistency"] >= 0.90)
            frame_stats[str(index)] = {
                "supported_roi_pixels": float(valid.sum()),
                "mean_consistency": float((maps["consistency"] * valid).sum() / valid_sum),
                "mean_reliability": float((maps["reliability"] * valid).sum() / valid_sum),
                "mean_disorder": float((maps["disorder"] * valid).sum() / valid_sum),
                "low_fraction": float(low.sum() / valid_sum),
                "medium_fraction": float(medium.sum() / valid_sum),
                "high_fraction": float(high.sum() / valid_sum),
            }

    valid_gaussians = reliability >= args.minimum_planarity
    summary = {
        "prepared_dir": str(prepared),
        "roi_dir": str(roi_dir),
        "neighbors": args.neighbors,
        "minimum_planarity": args.minimum_planarity,
        "valid_gaussian_fraction": float(valid_gaussians.mean()),
        "gaussian_consistency_quantiles": {
            str(q): float(np.quantile(consistency[valid_gaussians], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "frame_stats": frame_stats,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote Gaussian orientation consistency visualization to {output}")


if __name__ == "__main__":
    main()
