#!/usr/bin/env python3
"""Generate the frozen Trajectory orbit used by the E1 reconstruction.

Defaults are the dev-traj-v1 settings: 832x480, 50 steps, seed 20261030.
Finished outputs are left in place. The public loader in this repository reads
the Trajectory checkpoint; pass the Trajectory-480x832 weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggtdiff import VGGTDiff  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Trajectory orbit for E1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--omega-checkpoint", required=True)
    parser.add_argument("--base-model-dir", default=None)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--vram-limit-gib",
        type=float,
        default=46.0,
        help="Persistent CUDA weight budget. The frozen run used 46; use -1 to disable offload.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    result_path = args.output / "result.json"
    if result_path.exists():
        print("TRAJECTORY_EXISTS", args.output, flush=True)
        return
    source_paths = sorted((args.example / "source_views").glob("*.png"))
    if len(source_paths) != 6:
        raise ValueError(f"Expected six source images, found {len(source_paths)}")
    images = [Image.open(path).convert("RGB") for path in source_paths]
    trajectory_path = args.example / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    intrinsics_image_size = trajectory.get("intrinsics_image_size")
    if intrinsics_image_size is not None:
        expected_size = tuple(int(value) for value in intrinsics_image_size)
        if expected_size != images[0].size:
            raise ValueError(
                "trajectory intrinsics use image size "
                f"{expected_size}, but source images are {images[0].size}"
            )
    w2c = np.asarray(trajectory["source_w2c"] + trajectory["target_w2c"], dtype=np.float32)
    intrinsics = np.asarray(
        trajectory["source_intrinsics"] + trajectory["target_intrinsics"], dtype=np.float32
    )
    model = VGGTDiff(
        checkpoint=args.checkpoint,
        omega_checkpoint=args.omega_checkpoint,
        base_model_dir=args.base_model_dir,
        vram_limit_gib=None if args.vram_limit_gib < 0 else args.vram_limit_gib,
    )
    predictions = model.generate(
        images,
        w2c,
        intrinsics,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
    )
    frame_dir = args.output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, image in enumerate(predictions):
        path = frame_dir / f"{index:02d}.png"
        image.save(path)
        saved.append(dict(index=index, path=str(path.resolve()), sha256=sha256(path)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trajectory.json").write_text(trajectory_path.read_text())
    imageio.mimsave(
        args.output / "prediction.mp4",
        [np.asarray(image) for image in predictions],
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    result_path.write_text(
        json.dumps(
            dict(
                frames=saved,
                seed=args.seed,
                steps=args.steps,
                height=args.height,
                width=args.width,
                family="trajectory",
                checkpoint=args.checkpoint,
            ),
            indent=2,
        )
        + "\n"
    )
    print("TRAJECTORY_DONE", args.output, "frames", len(saved), flush=True)


if __name__ == "__main__":
    main()
