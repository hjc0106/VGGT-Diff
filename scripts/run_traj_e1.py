#!/usr/bin/env python3
"""Generate a Trajectory orbit, then train the E1 reconstruction.

The two stages can use different interpreters. Generation needs this package
and a CUDA GPU. Reconstruction also needs gsplat and a 3DGRUT checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trajectory generation and E1 reconstruction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--omega-checkpoint", required=True)
    parser.add_argument("--base-model-dir", default=None)
    parser.add_argument("--example", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--prompt", default="white car")
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--sam3-root")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--init-ply", type=Path)
    parser.add_argument("--threedgrut", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--vram-limit-gib", type=float, default=46.0)
    parser.add_argument("--generate-python", default=sys.executable)
    parser.add_argument("--reconstruct-python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video:
        work = args.work or (args.output / "preprocess")
        command = [
            args.generate_python, "-m", "gen_pro", "run",
            "--work", str(work),
            "--video", str(args.video),
            "--prompt", args.prompt,
            "--omega-checkpoint", args.omega_checkpoint,
            "--colmap", args.colmap,
            "--until", "orbit",
        ]
        if args.sam3_checkpoint:
            command.extend(["--sam3-checkpoint", args.sam3_checkpoint])
        if args.sam3_root:
            command.extend(["--sam3-root", args.sam3_root])
        subprocess.check_call(command, cwd=ROOT)
        args.example = work / "scene"
        args.init_ply = args.example / "initial_points.ply"
    if args.example is None or args.init_ply is None:
        raise SystemExit("Pass --video, or both --example and --init-ply")
    generated = args.output / "trajectory"
    reconstruction = args.output / "e1"
    generate = [
        args.generate_python,
        str(ROOT / "scripts" / "generate_trajectory.py"),
        "--checkpoint",
        args.checkpoint,
        "--omega-checkpoint",
        args.omega_checkpoint,
        "--example",
        str(args.example),
        "--output",
        str(generated),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--vram-limit-gib",
        str(args.vram_limit_gib),
    ]
    if args.base_model_dir:
        generate.extend(["--base-model-dir", args.base_model_dir])
    subprocess.check_call(generate)
    masks = args.masks
    if masks is None:
        if not args.video:
            raise SystemExit("Pass --masks, or --video so the generated frames can be segmented")
        work = args.work or (args.output / "preprocess")
        mask_command = [
            args.generate_python, "-m", "gen_pro", "frame-masks",
            "--work", str(work),
            "--prompt", args.prompt,
            "--generated-frames", str(generated / "frames"),
        ]
        if args.sam3_checkpoint:
            mask_command.extend(["--sam3-checkpoint", args.sam3_checkpoint])
        if args.sam3_root:
            mask_command.extend(["--sam3-root", args.sam3_root])
        try:
            subprocess.check_call(mask_command, cwd=ROOT)
        except subprocess.CalledProcessError as exc:
            review = work / "6_frame_masks" / "overrides.json"
            raise SystemExit(
                f"Frame masks need a manual choice. Set candidates in {review} and rerun."
            ) from exc
        masks = work / "6_frame_masks" / "masks"
    subprocess.check_call(
        [
            args.reconstruct_python,
            str(ROOT / "scripts" / "reconstruct_e1.py"),
            "--frames",
            str(generated / "frames"),
            "--trajectory",
            str(generated / "trajectory.json"),
            "--masks",
            str(masks),
            "--init-ply",
            str(args.init_ply),
            "--output",
            str(reconstruction),
            "--threedgrut",
            str(args.threedgrut),
        ]
    )


if __name__ == "__main__":
    main()
