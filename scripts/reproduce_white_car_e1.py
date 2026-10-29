#!/usr/bin/env python3
"""Reproduce the old white-car E1 from its frozen, hash-checked inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggtdiff.recon.benchmark import compare_generation, reproduce, verify_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=ROOT / "examples/white_car")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/white_car_e1_v1")
    parser.add_argument("--threedgrut", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Check every frozen input without CUDA")
    parser.add_argument("--generated-frames", type=Path,
                        help="Also require a regenerated 80-frame orbit to match the frozen pixels")
    args = parser.parse_args()
    if not (args.example / "frames/00.png").is_file():
        parser.error("Full frozen fixture is not bundled; pass --example /path/to/white_car_fixture (see examples/white_car/README.md)")
    manifest = verify_example(args.example)
    if args.generated_frames is not None:
        comparison = compare_generation(args.example, args.generated_frames)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "generation_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        if not comparison["passed"]:
            raise SystemExit("Regenerated pixels differ from the frozen run; do not reuse its masks. "
                             "See generation_comparison.json; use the frozen frames for strict E1 reproduction.")
        print("GENERATION_PIXELS_MATCH", len(comparison["frames"]), "frames")
    if args.verify_only:
        print("BENCHMARK_INPUTS_OK", len(manifest["files"]), "files")
        return
    if args.threedgrut is None:
        parser.error("--threedgrut is required unless --verify-only is used")
    reproduce(args.example, args.output, args.threedgrut)


if __name__ == "__main__":
    main()
