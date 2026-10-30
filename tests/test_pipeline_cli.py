"""The profile switch persists across preprocessing and reaches the full pipeline."""

import sys
import json
from dataclasses import asdict

import pytest

from gen_pro import cli
from gen_pro.config import Config
from gen_pro.sfm_profiles import profile_settings
from scripts import run_traj_reconstruction as runner


def test_profile_switch_keeps_saved_cuda_runtime_and_changes_ba_schedule(tmp_path, monkeypatch):
    observed = []

    def video(_workspace, cfg):
        observed.append((asdict(cfg), profile_settings(cfg.sfm_profile, cfg.sfm_final_ba_iterations)))
        return {}

    monkeypatch.setattr(cli, "PIPELINE", {"video": video})
    base = ["run", "--work", str(tmp_path), "--until", "video"]
    runtime = tmp_path / "isolated/bin/python"
    cli.main(base + ["--colmap-python", str(runtime), "--sfm-device", "cuda",
                     "--colmap", "/custom/colmap", "--sfm-profile", "reference",
                     "--sfm-keyframe-min-register-ratio", "0.96"])
    cli.main(base + ["--sfm-profile", "balanced"])
    cli.main(base)
    cli.main(base + ["--sfm-profile", "reference"])

    assert [cfg["sfm_profile"] for cfg, _settings in observed] == [
        "reference", "balanced", "balanced", "reference",
    ]
    assert [settings["final_ba_iterations"] for _cfg, settings in observed] == [0, 100, 100, 0]
    assert [settings["ba_global_frames_ratio"] for _cfg, settings in observed] == [1.1, 1.3, 1.3, 1.1]
    for cfg, _settings in observed:
        assert cfg["colmap_python"] == str(runtime)
        assert cfg["sfm_device"] == "cuda"
        assert cfg["sfm_ba_device"] == "cpu"
        assert cfg["colmap_bin"] == "/custom/colmap"
        assert cfg["sfm_keyframe_min_register_ratio"] == 0.96
    assert Config.load(tmp_path / "config.json").sfm_profile == "reference"


@pytest.mark.parametrize("profile,pose_refine", [("reference", False), ("balanced", True)])
def test_full_pipeline_forwards_profile_and_optional_pose_refine(tmp_path, monkeypatch, profile, pose_refine):
    commands = []
    monkeypatch.setattr(runner.subprocess, "check_call", lambda command, **_kwargs: commands.append(command))
    args = ["run_traj_reconstruction.py", "--video", str(tmp_path / "input.mp4"),
            "--output", str(tmp_path / "output"), "--checkpoint", "trajectory.safetensors",
            "--omega-checkpoint", "omega.pt", "--threedgrut", "/3dgrut",
            "--colmap-python", "/isolated/python", "--sfm-device", "cuda",
            "--sfm-profile", profile, "--sfm-keyframe-min-register-ratio", "0.96"]
    if pose_refine:
        args.append("--pose-refine")
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    preprocess, generate, masks, reconstruct = commands
    assert preprocess[preprocess.index("--sfm-profile") + 1] == profile
    assert preprocess[preprocess.index("--sfm-device") + 1] == "cuda"
    assert preprocess[preprocess.index("--colmap-python") + 1] == "/isolated/python"
    assert preprocess[preprocess.index("--sfm-keyframe-min-register-ratio") + 1] == "0.96"
    assert "--colmap" not in preprocess
    assert "--sfm-profile" not in generate and "--sfm-profile" not in masks
    assert ("--pose-refine" in reconstruct) == pose_refine
    assert reconstruct[1].endswith("/scripts/reconstruction.py")


def test_changed_trajectory_refuses_cached_generation(tmp_path):
    scene, generated = tmp_path / "scene", tmp_path / "generated"
    scene.mkdir()
    generated.mkdir()
    (generated / "result.json").write_text("{}")
    (generated / "trajectory.json").write_text(json.dumps({"target_w2c": ["old SfM"]}))
    (scene / "trajectory.json").write_text(json.dumps({"target_w2c": ["new SfM"]}))
    with pytest.raises(SystemExit, match="Choose a new --output"):
        runner.validate_existing_trajectory(scene, generated)


def test_same_trajectory_can_reuse_generation_when_resuming_masks(tmp_path):
    scene, generated = tmp_path / "scene", tmp_path / "generated"
    scene.mkdir()
    generated.mkdir()
    (generated / "result.json").write_text("{}")
    trajectory = {"target_w2c": [[1, 2, 3]], "source_w2c": [[4, 5, 6]]}
    (generated / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    (scene / "trajectory.json").write_text(json.dumps(trajectory))
    runner.validate_existing_trajectory(scene, generated)
