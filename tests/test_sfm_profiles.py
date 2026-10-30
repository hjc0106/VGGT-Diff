"""Schedule compatibility, keyframe coverage, and partial-registration parsing."""

import pytest

from gen_pro.colmap_io import _read_images
from gen_pro.sfm_profiles import keyframe_budgets, profile_settings, select_keyframes


def test_reference_preserves_existing_schedule_and_balanced_adds_final_ba():
    reference = profile_settings("reference")
    balanced = profile_settings("balanced")
    assert reference["ba_global_frames_ratio"] == 1.1
    assert reference["ba_global_max_refinements"] == 5
    assert reference["final_ba_iterations"] == 0
    assert balanced["ba_global_frames_ratio"] == balanced["ba_global_points_ratio"] == 1.3
    assert balanced["ba_global_max_refinements"] == 2
    assert balanced["final_ba_iterations"] == 100
    assert profile_settings("balanced", 40)["final_ba_iterations"] == 40
    with pytest.raises(ValueError):
        profile_settings("balanced", -1)


def test_keyframes_keep_required_views_endpoints_and_temporal_coverage():
    rows = [dict(name=f"{i:03d}.jpg", video_frame=i * 30, sharpness=i % 5) for i in range(125)]
    required = {"014.jpg", "015.jpg", "016.jpg", "063.jpg", "088.jpg", "100.jpg"}
    selected = select_keyframes(rows, 60, required)
    assert len(selected) == len(set(selected)) == 60
    assert required <= set(selected)
    assert selected[0] == "000.jpg" and selected[-1] == "124.jpg"
    indices = [int(name[:3]) for name in selected]
    assert max(b - a for a, b in zip(indices, indices[1:])) <= 4
    expanded = select_keyframes(rows, 90, set(selected))
    assert set(selected) <= set(expanded)
    assert len(expanded) == 90


def test_fallback_budgets_expand_to_full_input_without_duplicate_attempts():
    assert keyframe_budgets(125, 60) == [60, 90, 125]
    assert keyframe_budgets(80, 60) == [60, 80]
    assert keyframe_budgets(20, 60) == [20]


def test_images_without_point_observations_do_not_shift_following_poses(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text(
        "# Image list\n1 1 0 0 0 0 0 0 1 first.jpg\n\n"
        "2 1 0 0 0 1 2 3 1 second.jpg\n10 20 -1\n"
    )
    images = _read_images(path)
    assert [row["name"] for row in images] == ["first.jpg", "second.jpg"]
    assert images[1]["w2c"][0][3] == 1
