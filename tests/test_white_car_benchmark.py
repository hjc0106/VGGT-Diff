"""Integrity and acceptance checks for the actual portable E1 fixture. No CUDA."""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggtdiff.recon.benchmark import compare_metrics, read, verify_example


def test_frozen_example_has_disjoint_real_validation_and_original_orbit():
    example = ROOT / "examples/white_car"
    if not (example / "frames/00.png").exists():
        pytest.skip("Full frozen fixture is retained locally, not bundled")
    manifest = verify_example(example)
    cameras = read(example / "evaluation/cameras.json")
    assert manifest["source_frames"] == [1481, 1240, 937, 635, 272, 2630]
    assert len(cameras["heldout_real"]) == 16
    assert len(cameras["orbit360"]) == 160
    assert not set(manifest["source_frames"]) & {row["video_frame"] for row in cameras["heldout_real"]}
    assert all(not Path(row["image"]).is_absolute() for row in cameras["heldout_real"])


def test_corrupted_frame_is_rejected_before_training(tmp_path):
    if not (ROOT / "examples/white_car/frames/00.png").exists():
        pytest.skip("Full frozen fixture is retained locally, not bundled")
    example = tmp_path / "white_car"
    shutil.copytree(ROOT / "examples/white_car", example)
    (example / "frames/37.png").write_bytes(b"corrupted input")
    with pytest.raises(ValueError, match="frames/37.png"):
        verify_example(example)


def test_higher_psnr_does_not_hide_a_silhouette_regression():
    manifest = read(ROOT / "examples/white_car/benchmark.json")
    actual = {category: dict(median=copy.deepcopy(manifest["reference"][category]))
              for category in ("training", "real")}
    assert compare_metrics(actual, manifest)["passed"]
    actual["training"]["median"]["foreground_psnr"] += 0.3
    actual["training"]["median"]["silhouette_iou"] -= 0.08
    report = compare_metrics(actual, manifest)
    assert not report["passed"]
    assert any(row["metric"] == "silhouette_iou" and not row["passed"] for row in report["checks"])
