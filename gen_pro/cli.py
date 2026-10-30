"""Run the preprocessing steps from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gen_pro.config import Config
from gen_pro.steps import cameras, frame_masks, orbit, select, sources, video
from gen_pro.workspace import Workspace

PIPELINE = {
    "video": video.run,
    "select": select.run,
    "sources": sources.run,
    "cameras": cameras.run,
    "orbit": orbit.run,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a video for Trajectory generation and E1")
    parser.add_argument("action", choices=["run", "status", "frame-masks"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--prompt", default="white car")
    parser.add_argument("--omega-checkpoint")
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--sam3-root")
    parser.add_argument("--colmap", default=None)
    parser.add_argument("--colmap-python", type=Path, help="Python with pycolmap-cuda12 installed")
    parser.add_argument("--sfm-device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sfm-gpu-index", help="COLMAP CUDA device index, relative to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--sfm-num-threads", type=int)
    parser.add_argument("--sfm-ba-backend", choices=["ceres", "caspar"])
    parser.add_argument(
        "--sfm-profile", choices=["reference", "balanced", "keyframe"],
        help="SfM schedule: reference (default), balanced (fewer global BAs + final BA), "
             "or experimental keyframe. Saved in config.json; omitted flags keep the saved profile.",
    )
    parser.add_argument("--sfm-keyframes", type=int)
    parser.add_argument("--sfm-keyframe-min-register-ratio", type=float)
    parser.add_argument("--sfm-final-ba-iterations", type=int)
    parser.add_argument("--sfm-ba-device", choices=["cpu", "cuda"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from", dest="start", choices=list(PIPELINE), default="video")
    parser.add_argument("--until", dest="stop", choices=list(PIPELINE), default="orbit")
    parser.add_argument("--frames", type=int, nargs="*")
    parser.add_argument("--generated-frames", type=Path)
    args = parser.parse_args(argv)
    workspace = Workspace(args.work)
    config_path = workspace.root / "config.json"
    cfg = Config.load(config_path) if config_path.exists() else Config()
    if args.video:
        cfg.video = str(args.video.resolve())
    cfg.prompt = args.prompt
    if args.omega_checkpoint:
        cfg.omega_checkpoint = args.omega_checkpoint
    if args.sam3_checkpoint:
        cfg.sam3_checkpoint = args.sam3_checkpoint
    if args.sam3_root:
        cfg.sam3_root = args.sam3_root
    if args.colmap:
        cfg.colmap_bin = args.colmap
    if args.colmap_python:
        cfg.colmap_python = str(args.colmap_python.absolute())
    for name in ("sfm_device", "sfm_gpu_index", "sfm_num_threads", "sfm_ba_backend",
                 "sfm_profile", "sfm_keyframes", "sfm_keyframe_min_register_ratio",
                 "sfm_final_ba_iterations", "sfm_ba_device"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    cfg.device = args.device
    if args.frames:
        cfg.manual_frames = tuple(args.frames)
    if args.generated_frames:
        cfg.generated_frames = str(args.generated_frames.resolve())
    cfg.save(config_path)
    if args.action == "status":
        for row in workspace.status():
            print(f"{row['step']:14} {row['status']}")
        return
    if args.action == "frame-masks":
        frame_masks.run(workspace, cfg)
        return
    names = list(PIPELINE)
    start = names.index(args.start)
    stop = names.index(args.stop)
    if stop < start:
        raise SystemExit("--until is before --from")
    for name in names[start : stop + 1]:
        metrics = PIPELINE[name](workspace, cfg)
        if name == "cameras":
            reaim(workspace, cfg, metrics)


def reaim(workspace, cfg, metrics) -> None:
    """Aim the references once more at the cloud center when the mask rays missed it."""
    overrides = workspace.overrides("sources")
    manual = overrides.get("center") is not None and overrides.get("center_source") != "cloud"
    if manual or metrics.get("aim_shift_ratio", 0.0) <= cfg.aim_refine_ratio:
        return
    print("REAIM", round(metrics["aim_shift_ratio"], 4), flush=True)
    workspace.write_overrides("sources", {**overrides, "center": metrics["cloud_center"], "center_source": "cloud"})
    sources.run(workspace, cfg)
    cameras.run(workspace, cfg)


if __name__ == "__main__":
    main()
