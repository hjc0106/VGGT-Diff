"""Frozen E1 schedule, orbit holdout, and dataset preparation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggtdiff.recon.colmap import rotation_to_quat_wxyz
from vggtdiff.recon.e1 import prepare, training_indices


def test_closed_orbit_holds_out_the_last_frame():
    poses = np.tile(np.eye(4), (80, 1, 1))
    poses[-1] = poses[0]
    assert training_indices(poses) == list(range(79))


def test_seventy_nine_views_are_kept():
    assert training_indices(np.tile(np.eye(4), (79, 1, 1))) == list(range(79))


def test_schedule_matches_the_trained_e1_run():
    defaults = json.loads((ROOT / "vggtdiff/recon/e1_defaults.json").read_text())
    schedule = defaults["schedule"]
    assert defaults["updates_per_image"] == 625
    assert defaults["outside_weight"] == 0.5
    assert schedule["position_max_steps"] == 79 * 625
    assert schedule["densify_frequency"] == 2961
    assert schedule["densify_start"] == 4934
    assert schedule["densify_end"] == 29605
    assert schedule["prune_frequency"] == 987
    assert schedule["prune_end"] == 44408
    assert schedule["increase_frequency"] == 9868
    assert schedule["checkpoints"] == [16447, 32895, 49375]


def test_config_template_has_no_machine_paths():
    text = (ROOT / "vggtdiff/recon/e1_config.yaml").read_text()
    assert "/home/" not in text
    assert "n_iterations: 49375" in text


def test_identity_rotation_is_a_unit_quaternion():
    quat = rotation_to_quat_wxyz(np.eye(3))
    assert np.allclose(quat, [1, 0, 0, 0])


def test_prepare_holds_out_the_closure_frame():
    defaults = json.loads((ROOT / "vggtdiff/recon/e1_defaults.json").read_text())
    vertices = defaults["real_surface_points"] + defaults["volume_seed_points"]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frames, masks = root / "frames", root / "masks"
        frames.mkdir()
        masks.mkdir()
        image = Image.new("RGB", tuple(defaults["image_size"]), "white")
        mask = Image.new("L", tuple(defaults["image_size"]), 255)
        poses = []
        intrinsics = []
        for index in range(80):
            image.save(frames / f"{index:02d}.png")
            mask.save(masks / f"{index:02d}.png")
            pose = np.eye(4)
            pose[0, 3] = index
            poses.append(pose.tolist())
            intrinsics.append(np.eye(3).tolist())
        poses[-1] = poses[0]
        trajectory = root / "trajectory.json"
        trajectory.write_text(
            json.dumps(dict(target_w2c=poses, target_intrinsics=intrinsics))
        )
        ply = root / "initial_points.ply"
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {vertices}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        ply.write_bytes(header.encode() + b"\0" * (vertices * 12))
        output = root / "e1"
        experiment = prepare(output, frames, trajectory, masks, ply)
        assert experiment["images"] == 79
        assert experiment["steps"] == 49375
        assert experiment["frames"] == list(range(79))
        assert (output / "dataset/images/78.png").is_file()
        assert not (output / "dataset/images/79.png").exists()
        assert (output / "dataset/images/00_mask.png").is_file()
        plan = json.loads((output / "sampling_plan.json").read_text())
        assert len(plan["stems"]) == 49375
        again = prepare(output, frames, trajectory, masks, ply)
        assert again["steps"] == experiment["steps"]


if __name__ == "__main__":
    test_closed_orbit_holds_out_the_last_frame()
    test_seventy_nine_views_are_kept()
    test_schedule_matches_the_trained_e1_run()
    test_config_template_has_no_machine_paths()
    test_identity_rotation_is_a_unit_quaternion()
    test_prepare_holds_out_the_closure_frame()
    print("OK")
