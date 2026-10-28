#!/usr/bin/env python3
"""Train the frozen E1 3DGS reconstruction from a generated orbit.

Masks are indexed like the generated frames (`00.png` for frame 0). An 80-frame
orbit whose first and last cameras match holds out that closure frame. Training
imports 3DGRUT from `--threedgrut` and rasterizes with gsplat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggtdiff.recon.e1 import reconstruct  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct E1 from Trajectory frames")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--init-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threedgrut", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruct(
        args.output,
        args.frames,
        args.trajectory,
        args.masks,
        args.init_ply,
        args.threedgrut,
    )
    print("E1_DONE", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
