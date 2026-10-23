import unittest

import numpy as np

from crossview_geometry_eval import (
    apply_intrinsics,
    build_tracks,
    camera_center,
    fundamental_matrix,
    heldout_attempts,
    pair_record,
    perturb_center,
    perturb_rotation,
    project_point,
    relative_pose,
    sampson_sqrt_px,
    transform_points,
    triangulate_point,
)


def intrinsics():
    return np.array([[800.0, 0.0, 416.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])


def pose_at(center):
    pose = np.eye(4)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64)
    return pose


def pixels(k, pose, point):
    projected, depth = project_point(k, pose, point)
    if projected is None:
        raise AssertionError(f'point is not in front of the camera, depth {depth}')
    return projected


class AnalyticGeometryTests(unittest.TestCase):
    def setUp(self):
        self.k = intrinsics()
        self.point = np.array([0.15, -0.05, 5.0])
        self.poses = [pose_at(center) for center in ((0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.0, 0.25, 0.0))]
        self.observed = [pixels(self.k, pose, self.point) for pose in self.poses]
        self.mask = np.full((480, 832), 255, np.uint8)

    def test_projection_and_sampson_are_numerically_zero(self):
        fundamental, translation = fundamental_matrix(self.k, self.poses[0], self.k, self.poses[1])
        self.assertGreater(np.linalg.norm(translation), 0.1)
        error = sampson_sqrt_px(fundamental, self.observed[0][None], self.observed[1][None])
        self.assertLess(error[0], 1e-6)
        recovered = triangulate_point(self.k, self.poses[0], self.observed[0], self.k, self.poses[1], self.observed[1])
        np.testing.assert_allclose(recovered, self.point, atol=1e-8)

    def test_sampson_sqrt_matches_a_perpendicular_pixel_shift(self):
        fundamental, _ = fundamental_matrix(self.k, self.poses[0], self.k, self.poses[1])
        line = fundamental @ np.array([self.observed[0][0], self.observed[0][1], 1.0])
        normal = line[:2] / np.linalg.norm(line[:2])
        shifted = self.observed[1] + 3.0 * normal
        error = sampson_sqrt_px(fundamental, self.observed[0][None], shifted[None])
        # One-sided shift of d pixels. With similar epipolar gradients the symmetric
        # Sampson correction is d/sqrt(2), not d and not d squared.
        self.assertAlmostEqual(error[0], 3.0 / np.sqrt(2.0), delta=0.05)

    def test_intrinsics_scale_normalizes_back_to_the_original_grid(self):
        fundamental, _ = fundamental_matrix(self.k, self.poses[0], self.k, self.poses[1])
        line = fundamental @ np.array([self.observed[0][0], self.observed[0][1], 1.0])
        normal = line[:2] / np.linalg.norm(line[:2])
        shifted = self.observed[1] + 2.0 * normal
        scale = np.diag([1.5, 1.5, 1.0])
        scaled_k = apply_intrinsics(self.k, scale)
        scaled_i, _ = transform_points(self.observed[0][None], scale)
        scaled_j, _ = transform_points(shifted[None], scale)
        scaled_fundamental, _ = fundamental_matrix(scaled_k, self.poses[0], scaled_k, self.poses[1])
        original = sampson_sqrt_px(fundamental, self.observed[0][None], shifted[None])[0]
        scaled_error = sampson_sqrt_px(scaled_fundamental, scaled_i, scaled_j)[0] / 1.5
        self.assertAlmostEqual(scaled_error, original, delta=1e-6)
        yawed = perturb_rotation(self.poses[1], np.array([0.0, 1.0, 0.0]), 8.0)
        observed_yaw = pixels(self.k, yawed, self.point)
        scaled_yaw, _ = transform_points(observed_yaw[None], scale)
        correct, _ = fundamental_matrix(scaled_k, self.poses[0], scaled_k, yawed)
        stale, _ = fundamental_matrix(self.k, self.poses[0], self.k, yawed)
        self.assertLess(sampson_sqrt_px(correct, scaled_i, scaled_yaw)[0], 1e-6)
        self.assertGreater(sampson_sqrt_px(stale, scaled_i, scaled_yaw)[0], 2.0)

    def test_correct_affine_keeps_perfect_correspondences(self):
        affine = np.array([[1.1, 0.0, -20.0], [0.0, 1.2, -15.0], [0.0, 0.0, 1.0]])
        updated = apply_intrinsics(self.k, affine)
        moved = [transform_points(pixel[None], affine)[0][0] for pixel in self.observed]
        fundamental, _ = fundamental_matrix(updated, self.poses[0], updated, self.poses[1])
        error = sampson_sqrt_px(fundamental, moved[0][None], moved[1][None])
        self.assertLess(error[0], 1e-6)

    def test_relative_pose_direction_and_wrong_rotation(self):
        rotation, translation = relative_pose(self.poses[0], self.poses[1])
        camera_point_i = self.poses[0][:3, :3] @ self.point + self.poses[0][:3, 3]
        camera_point_j = rotation @ camera_point_i + translation
        expected = self.poses[1][:3, :3] @ self.point + self.poses[1][:3, 3]
        np.testing.assert_allclose(camera_point_j, expected, atol=1e-8)
        rolled = perturb_rotation(self.poses[1], np.array([0.0, 0.0, 1.0]), 20.0)
        fundamental, _ = fundamental_matrix(self.k, self.poses[0], self.k, rolled)
        error = sampson_sqrt_px(fundamental, self.observed[0][None], self.observed[1][None])
        self.assertGreater(error[0], 5.0)

    def test_zero_translation_is_degenerate_and_not_zero_error(self):
        record = pair_record(0, 1, self.observed[0][None], self.observed[1][None], self.mask, self.mask, self.k, self.poses[0], self.k, self.poses[0])
        self.assertTrue(record['degenerate_translation'])
        self.assertIsNone(record['epi_median_px'])

    def test_missing_matches_are_insufficient_evidence(self):
        record = pair_record(0, 1, np.zeros((0, 2)), np.zeros((0, 2)), self.mask, self.mask, self.k, self.poses[0], self.k, self.poses[1])
        self.assertEqual(record['evidence'], 'insufficient')
        self.assertIsNone(record['epi_median_px'])
        self.assertNotEqual(record['epi_median_px'], 0)

    def test_negative_depth_and_low_parallax_are_separate_failures(self):
        behind = np.array([0.15, -0.05, -5.0])
        observations = []
        for pose in self.poses[:2]:
            camera_point = pose[:3, :3] @ behind + pose[:3, 3]
            pixel = self.k @ camera_point
            observations.append(pixel[:2] / pixel[2])
        recovered = triangulate_point(self.k, self.poses[0], observations[0], self.k, self.poses[1], observations[1])
        depth = (self.poses[0][:3, :3] @ recovered + self.poses[0][:3, 3])[2]
        self.assertLess(depth, 0.0)
        close = pose_at((0.0001, 0.0, 0.0))
        frames = self._frames([self.poses[0], close, self.poses[2]], self.observed)
        frames[1]['xy'] = pixels(self.k, close, self.point)[None]
        attempts = heldout_attempts(dict(track_id=1, conflict=False, n_views=3, observations={0: 0, 1: 0, 2: 0}), frames, 1.0)
        held_close = next(row for row in attempts if row['heldout_frame'] == 2)
        self.assertEqual(held_close['reason'], 'low_parallax')

    def test_heldout_view_does_not_enter_the_fit(self):
        frames = self._frames(self.poses, self.observed)
        frames[2]['xy'] = (self.observed[2] + np.array([5.0, 0.0]))[None]
        attempts = heldout_attempts(dict(track_id=3, conflict=False, n_views=3, observations={0: 0, 1: 0, 2: 0}), frames, 1.0)
        held = next(row for row in attempts if row['heldout_frame'] == 2)
        self.assertEqual(held['reason'], 'ok')
        self.assertAlmostEqual(held['error_px'], 5.0, delta=0.05)
        self.assertEqual({held['fit_a'], held['fit_b']}, {0, 1})

    def test_conflicting_track_is_rejected(self):
        matches = {(0, 1): (np.array([0, 0]), np.array([0, 1]))}
        tracks = build_tracks(matches)
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0]['conflict'])

    def test_camera_perturbations_move_only_the_requested_component(self):
        rotated = perturb_rotation(self.poses[1], np.array([0.0, 1.0, 0.0]), 1.0)
        np.testing.assert_allclose(camera_center(rotated), camera_center(self.poses[1]), atol=1e-8)
        moved = perturb_center(self.poses[1], np.array([0.0, 0.0, 1.0]), 0.2)
        np.testing.assert_allclose(camera_center(moved) - camera_center(self.poses[1]), [0.0, 0.0, 0.2], atol=1e-8)
        np.testing.assert_allclose(moved[:3, :3], self.poses[1][:3, :3])

    def _frames(self, poses, observed):
        frames = {}
        for index, (pose, pixel) in enumerate(zip(poses, observed)):
            frames[index] = dict(K=self.k, T=pose, xy=np.asarray(pixel, dtype=np.float64)[None], mask=self.mask)
        return frames


if __name__ == '__main__':
    unittest.main()
