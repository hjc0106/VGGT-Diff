"""Geometry and workspace tests for gen_pro. No GPU."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gen_pro.omega_dense import intrinsics_to_image, maps_to_image
from gen_pro.steps.cameras import _load_cache, depth_to_world, projection_hit
from gen_pro.steps.frame_masks import area_jumps
from gen_pro.steps.select import choose_references
from gen_pro.steps.sources import aim_view, mask_ray
from gen_pro.geometry import (
    apply_similarity,
    camera_center,
    continuous_orbit,
    framing_margin,
    look_at,
    ray_convergence,
    transform_w2c,
    umeyama,
)
from gen_pro.steps.orbit import map_reference_targets
from gen_pro.workspace import Workspace


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(12, 3))
    rotation = _rotation_about(np.array([0.2, -0.4, 0.9]), 0.7)
    transformed = apply_similarity(source, 2.5, rotation, np.array([1.0, -2.0, 0.5]))
    scale, estimated, translation = umeyama(source, transformed)
    assert abs(scale - 2.5) < 1e-8
    assert np.allclose(estimated, rotation, atol=1e-8)
    assert np.allclose(translation, [1, -2, 0.5], atol=1e-8)


def test_rays_meet_at_the_known_center():
    center = np.array([0.2, -0.1, 0.4])
    origins = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.3], [-1.0, -1.0, 1.5], [0.4, -2.0, 0.2]])
    directions = center - origins
    point, distance = ray_convergence(origins, directions)
    assert np.allclose(point, center, atol=1e-8)
    assert np.max(distance) < 1e-8


def test_orbit_closes_and_keeps_radius():
    center = np.zeros(3)
    up = np.array([0.0, 0.0, 1.0])
    anchor = look_at(np.array([2.0, 0.0, 0.5]), center, up)
    intrinsics = np.array([[800.0, 0, 416.0], [0, 800.0, 240.0], [0, 0, 1]])
    angles = np.linspace(0, 360, 80)
    orbit = continuous_orbit(anchor, intrinsics, 0.0, center, up, angles)
    poses = np.asarray(orbit["target_w2c"])
    assert np.allclose(poses[0], poses[-1])
    centers = np.array([camera_center(pose) for pose in poses])
    radius = np.linalg.norm(centers[:, :2], axis=1)
    assert np.ptp(radius) < 1e-8
    assert np.ptp(centers[:, 2]) < 1e-8


def test_reference_targets_follow_the_source_similarity():
    rng = np.random.default_rng(1)
    up = np.array([0.0, 0.0, 1.0])
    sources = [look_at(rng.normal(size=3) * 2, np.zeros(3), up) for _ in range(6)]
    targets = [look_at(rng.normal(size=3) * 2, np.zeros(3), up) for _ in range(4)]
    reference = dict(
        source_frames=[30, 10, 50, 70, 90, 0],
        source_w2c=[pose.tolist() for pose in sources],
        target_w2c=[pose.tolist() for pose in targets],
    )
    rotation = _rotation_about(np.array([0.3, 0.1, 1.0]), 0.9)
    move = lambda pose: transform_w2c(pose, 0.4, rotation, np.array([0.5, -1.0, 2.0]))
    order = [10, 0, 30, 90, 50, 70]
    by_frame = dict(zip(reference["source_frames"], sources))
    positions = np.array([camera_center(move(by_frame[frame])) for frame in order])
    poses, fit = map_reference_targets(reference, order, positions)
    assert np.allclose(poses, [move(pose) for pose in targets], atol=1e-8)
    assert abs(fit["scale"] - 0.4) < 1e-8
    assert max(fit["center_residual_ratio"]) < 1e-8


def test_framing_margin_of_a_centered_box():
    center = np.zeros(3)
    pose = look_at(np.array([0.0, 0.0, 3.0]), center, np.array([0.0, 1.0, 0.0]))
    intrinsics = np.array([[500.0, 0, 416.0], [0, 500.0, 240.0], [0, 0, 1]])
    corners = np.array([[x, y, z] for x in (-0.2, 0.2) for y in (-0.1, 0.1) for z in (-0.1, 0.1)])
    margin = framing_margin(corners, pose, intrinsics, (832, 480))
    assert margin > 40


def test_reference_selection_spreads_azimuth_and_accepts_manual_frames():
    rows = [
        dict(video_frame=frame, registered=True, sharpness=10 + frame, azimuth_deg=float(frame))
        for frame in range(0, 100, 10)
    ]
    chosen, reason, _threshold = choose_references(rows, 6, 0.0, [])
    assert reason == "azimuth"
    assert [row["video_frame"] for row in chosen] == [0, 20, 40, 50, 70, 90]
    manual, reason, _threshold = choose_references(rows, 6, 0.0, [30, 10, 50, 70, 90, 0])
    assert reason == "manual"
    assert [row["video_frame"] for row in manual] == [30, 10, 50, 70, 90, 0]


def test_rerun_marks_downstream_stale_and_overrides_stick():
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Workspace(Path(temporary))
        workspace.complete("video", {"a": 1}, [], {"registered": 1})
        workspace.complete("select", {"b": 2}, [], {})
        workspace.complete("video", {"a": 1}, [], {"registered": 1})
        assert workspace.record("select")["status"] == "stale"
        workspace.write_overrides("select", {"frames": [1, 2, 3, 4, 5, 6]})
        assert workspace.overrides("select")["frames"] == [1, 2, 3, 4, 5, 6]
        assert workspace.fresh("video", {"a": 1})
        assert not workspace.fresh("video", {"a": 2})


def _blob_view(point, pose, intrinsics, size=(3840, 2160), radius=30):
    pixel = intrinsics @ (pose[:3, :3] @ point + pose[:3, 3])
    u, v = pixel[:2] / pixel[2]
    yy, xx = np.mgrid[: size[1], : size[0]]
    return (xx - u) ** 2 + (yy - v) ** 2 < radius**2


def test_aimed_view_centers_the_car_at_the_shared_focal():
    intrinsics = np.array([[2132.0, 0, 1920], [0, 2135.0, 1080], [0, 0, 1]])
    pose = look_at(np.array([0.0, 0.0, -2.0]), np.array([0.5, 0.2, 0.0]), np.array([0.0, -1.0, 0.0]))
    center = np.array([0.1, 0.05, 0.0])
    mask = _blob_view(center, pose, intrinsics)
    image = np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2)
    view, view_mask, aimed, view_k, angle, reason = aim_view(
        image, mask, pose, intrinsics, center, np.array([0.0, -1.0, 0.0]), 965.0, 832, 480
    )
    assert reason is None and angle > 5
    assert view_k[0, 0] == 965.0
    assert np.allclose(camera_center(aimed), camera_center(pose))
    assert abs(aimed[0, :3] @ np.array([0.0, -1.0, 0.0])) < 1e-9
    pixel = view_k @ (aimed[:3, :3] @ center + aimed[:3, 3])
    assert np.allclose(pixel[:2] / pixel[2], [415.5, 239.5], atol=1e-6)
    ys, xs = np.where(view_mask > 127)
    assert abs(xs.mean() - 415.5) < 1.5 and abs(ys.mean() - 239.5) < 1.5
    assert view[240, 416].mean() > 200


def test_aimed_view_lowers_the_focal_only_when_the_car_does_not_fit():
    intrinsics = np.array([[2132.0, 0, 1920], [0, 2135.0, 1080], [0, 0, 1]])
    pose = np.eye(4)
    mask = np.zeros((2160, 3840), dtype=bool)
    mask[900:1260, 700:3140] = True
    image = np.zeros((2160, 3840, 3), np.uint8)
    _view, view_mask, _pose, view_k, _angle, reason = aim_view(
        image, mask, pose, intrinsics, np.array([0.0, 0.0, 5.0]), np.array([0.0, -1.0, 0.0]), 965.0, 832, 480
    )
    assert reason is not None and view_k[0, 0] < 965.0
    xs = np.where(view_mask > 127)[1]
    assert xs.min() >= 0.05 * 832 - 2 and xs.max() <= 0.95 * 832 + 2


def test_aimed_view_raises_the_focal_to_stay_inside_the_frame():
    intrinsics = np.array([[2132.0, 0, 1920], [0, 2135.0, 1080], [0, 0, 1]])
    pose = np.eye(4)
    center = np.array([np.tan(np.radians(35)) * 5, 0.0, 5.0])
    mask = _blob_view(center, pose, intrinsics, radius=60)
    image = np.full((2160, 3840, 3), 255, np.uint8)
    view, _mask, _pose, view_k, _angle, reason = aim_view(
        image, mask, pose, intrinsics, center, np.array([0.0, -1.0, 0.0]), 965.0, 832, 480
    )
    assert view_k[0, 0] > 965.0 and reason.startswith("focal raised")
    assert view.min() > 200


def test_mask_rays_meet_at_the_blob_center():
    intrinsics = np.array([[2132.0, 0, 1920], [0, 2135.0, 1080], [0, 0, 1]])
    center = np.array([0.2, -0.1, 0.3])
    rays = []
    for position in ([3.0, 0.0, 0.0], [0.0, 0.2, 3.0], [-2.0, 0.1, -2.0]):
        pose = look_at(np.array(position), center + 0.1, np.array([0.0, -1.0, 0.0]))
        rays.append(mask_ray(pose, intrinsics, _blob_view(center, pose, intrinsics)))
    point, _miss = ray_convergence(np.array([r[0] for r in rays]), np.array([r[1] for r in rays]))
    assert np.linalg.norm(point - center) < 2e-3


def test_omega_depth_is_cast_along_the_known_rays():
    intrinsics = np.array([[40.0, 0, 15.5], [0, 40.0, 11.5], [0, 0, 1]])
    pose = look_at(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.array([0.0, -1.0, 0.0]))
    depth = np.full((24, 32), 3.0)
    u, v = np.meshgrid(np.arange(32.0), np.arange(24.0))
    camera = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(intrinsics).T * depth[..., None]
    truth = (camera - pose[:3, 3]) @ pose[:3, :3]
    rotation = _rotation_about(np.array([0.0, 1.0, 0.0]), 0.4)
    omega_pose = transform_w2c(pose, 0.5, rotation, np.array([1.0, 0.0, 0.0]))
    omega_world = apply_similarity(truth.reshape(-1, 3), 0.5, rotation, np.array([1.0, 0.0, 0.0])).reshape(truth.shape)
    world = depth_to_world(omega_world[None], [omega_pose], 2.0, [pose], [intrinsics])
    assert np.allclose(world[0], truth, atol=1e-9)


def test_omega_intrinsics_include_the_centered_crop():
    omega_k = np.array([[100.0, 0, 50], [0, 110.0, 40], [0, 0, 1]])
    mapped = intrinsics_to_image(omega_k, (200, 40), (20, 40))
    assert abs(mapped[0, 0] - 200) < 1e-6
    assert abs(mapped[0, 2] - 160) < 1e-6
    placed = maps_to_image(np.ones((1, 2, 2), np.float32), (10, 4), (2, 2))
    assert np.isnan(placed[0, :, 0]).all()
    assert np.isfinite(placed[0, :, 1:9]).all()


def test_omega_cache_rejects_a_different_source_digest():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "omega.npz"
        np.savez(path, sources=np.array("aaa"), w2c=np.eye(4))
        assert _load_cache(path, "aaa")["w2c"].shape == (4, 4)
        assert _load_cache(path, "bbb") is None


def test_a_missing_frame_mask_fails_instead_of_copying_another():
    from gen_pro.sam3_masks import Segmenter
    from gen_pro.steps import frame_masks

    class EmptySegmenter:
        def __init__(self, *args, **kwargs):
            pass

        def candidates(self, image, prompt, confidence=None):
            if confidence is None:
                return []
            mask = np.zeros(image.size[::-1], dtype=bool)
            mask[0, 0] = True
            return [dict(id=0, score=0.02, area=1, fraction=0.01, box=[0, 0, 1, 1], mask=mask)]

    frame_masks.Segmenter = EmptySegmenter
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            frames.mkdir()
            Image.new("RGB", (8, 8)).save(frames / "00.png")
            workspace = Workspace(Path(tmp) / "work")

            class Config:
                sam3_checkpoint = "unused"
                sam3_root = None
                device = "cpu"
                sam3_confidence = 0.15
                mask_min_area = 0.005
                mask_max_area = 0.9
                prompt = "white car"
                generated_frames = str(frames)

            try:
                frame_masks.run(workspace, Config())
            except RuntimeError as exc:
                assert "overrides.json" in str(exc)
            else:
                raise AssertionError("a frame without a mask should fail the step")
            assert not (workspace.step_dir("frame_masks") / "masks" / "00.png").exists()
            assert workspace.record("frame_masks")["status"] == "failed"
    finally:
        frame_masks.Segmenter = Segmenter


def test_projection_hit_is_the_in_mask_fraction_of_visible_points():
    pose = np.eye(4)
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.2, 0.0, 2.0], [0.4, 0.0, 2.0], [0.0, 0.0, -1.0]])
    mask = np.zeros((80, 100), dtype=bool)
    mask[40, 60] = True
    assert projection_hit(points, pose, intrinsics, mask) == 0.5
    mask[40, 70] = True
    assert projection_hit(points, pose, intrinsics, mask) == 1.0
    shifted = intrinsics.copy()
    shifted[0, 0] = 250.0
    assert projection_hit(points, pose, shifted, mask) == 0.0


def test_mask_area_jump_is_detected():
    assert area_jumps([100, 400, 100], 1)
    assert not area_jumps([100, 200, 100], 1)


def _rotation_about(axis, radians):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine, sine = np.cos(radians), np.sin(radians)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


if __name__ == "__main__":
    test_umeyama_recovers_a_known_similarity()
    test_rays_meet_at_the_known_center()
    test_orbit_closes_and_keeps_radius()
    test_reference_targets_follow_the_source_similarity()
    test_framing_margin_of_a_centered_box()
    test_reference_selection_spreads_azimuth_and_accepts_manual_frames()
    test_rerun_marks_downstream_stale_and_overrides_stick()
    test_aimed_view_centers_the_car_at_the_shared_focal()
    test_aimed_view_lowers_the_focal_only_when_the_car_does_not_fit()
    test_aimed_view_raises_the_focal_to_stay_inside_the_frame()
    test_mask_rays_meet_at_the_blob_center()
    test_omega_depth_is_cast_along_the_known_rays()
    test_omega_intrinsics_include_the_centered_crop()
    test_omega_cache_rejects_a_different_source_digest()
    test_a_missing_frame_mask_fails_instead_of_copying_another()
    test_projection_hit_is_the_in_mask_fraction_of_visible_points()
    test_mask_area_jump_is_detected()
    print("OK")
