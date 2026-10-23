#!/usr/bin/env python3
"""Build a full-object ROI from an ArtiFixer opacity trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opacity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--close-radius", type=int, default=8)
    parser.add_argument("--dilate-radius", type=int, default=5)
    parser.add_argument("--edge-blur-radius", type=float, default=2.0)
    parser.add_argument("--replace", action="store_true")
    return parser


def ellipse(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def main() -> None:
    args = build_parser().parse_args()
    opacity_dir = args.opacity_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        raise FileExistsError(f"{output_dir} is not empty; pass --replace")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    overlay_dir = output_dir / "overlay"
    preview_dir.mkdir(exist_ok=True)
    if args.render_dir is not None:
        overlay_dir.mkdir(exist_ok=True)

    opacity_paths = sorted(opacity_dir.glob("*.png"))
    if not opacity_paths:
        raise FileNotFoundError(f"no opacity PNG files in {opacity_dir}")

    for opacity_path in opacity_paths:
        opacity = np.asarray(Image.open(opacity_path).convert("L"), dtype=np.float32) / 255.0
        binary = (opacity >= args.opacity_threshold).astype(np.uint8)

        # Close small gaps inside the rendered silhouette, then retain a narrow
        # halo so low-opacity missing structure remains inside the evaluation ROI.
        if args.close_radius > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ellipse(args.close_radius))
        flood = binary.copy()
        padded = np.pad(flood, 1, mode="constant", constant_values=0)
        fill = padded.copy()
        flood_mask = np.zeros((fill.shape[0] + 2, fill.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(fill, flood_mask, (0, 0), 1)
        holes = 1 - fill[1:-1, 1:-1]
        binary = np.maximum(binary, holes)
        if args.dilate_radius > 0:
            binary = cv2.dilate(binary, ellipse(args.dilate_radius))

        roi = binary.astype(np.float32)
        if args.edge_blur_radius > 0:
            roi = cv2.GaussianBlur(roi, (0, 0), args.edge_blur_radius)
            roi /= max(float(roi.max()), 1e-8)
        roi = np.clip(roi, 0.0, 1.0).astype(np.float32)

        stem = opacity_path.stem
        np.save(output_dir / f"{stem}.npy", roi)
        Image.fromarray(np.round(roi * 255.0).astype(np.uint8), mode="L").save(
            preview_dir / f"{stem}.png"
        )

        if args.render_dir is not None:
            render = np.asarray(
                Image.open(args.render_dir.resolve() / f"{stem}.png").convert("RGB"),
                dtype=np.float32,
            ) / 255.0
            color = np.array((0.15, 0.55, 1.0), dtype=np.float32)
            alpha = roi[..., None] * 0.42
            overlay = render * (1.0 - alpha) + color * alpha
            Image.fromarray(np.round(overlay * 255.0).astype(np.uint8), mode="RGB").save(
                overlay_dir / f"{stem}.png"
            )

    print(f"Wrote {len(opacity_paths)} full-object ROI masks to {output_dir}")


if __name__ == "__main__":
    main()
