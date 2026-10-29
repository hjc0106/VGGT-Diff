"""Camera and point-cloud geometry shared by the preprocessing steps."""

from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / length


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity taking source points onto target points: ``scale * R @ x + t``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Umeyama expects matching [N, 3] point sets")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean) / len(source)
    left, singular, right = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right
    variance = np.square(source - source_mean).sum() / len(source)
    scale = float((singular * sign).sum() / max(variance, 1e-12))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid similarity scale {scale}")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray):
    points = np.asarray(points, dtype=np.float64)
    return scale * (points @ rotation.T) + translation


def transform_w2c(w2c: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray):
    """Map a world-to-camera pose into the target frame of a similarity on world points."""
    w2c = np.asarray(w2c, dtype=np.float64)
    center = camera_center(w2c)
    mapped = apply_similarity(center, scale, rotation, translation)
    pose = np.eye(4)
    pose[:3, :3] = w2c[:3, :3] @ rotation.T
    pose[:3, 3] = -pose[:3, :3] @ mapped
    return pose


def camera_center(w2c: np.ndarray) -> np.ndarray:
    rotation = np.asarray(w2c, dtype=np.float64)[:3, :3]
    translation = np.asarray(w2c, dtype=np.float64)[:3, 3]
    return -rotation.T @ translation


def camera_axes(w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-space forward (+Z) and up (-Y) of an OpenCV camera."""
    rotation = np.asarray(w2c, dtype=np.float64)[:3, :3]
    forward = rotation.T @ np.array([0.0, 0.0, 1.0])
    up = rotation.T @ np.array([0.0, -1.0, 0.0])
    return forward, up


def ray_convergence(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares intersection of rays and the distance of each ray to that point."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    system = np.zeros((3, 3))
    target = np.zeros(3)
    for origin, direction in zip(origins, directions):
        projection = np.eye(3) - np.outer(direction, direction)
        system += projection
        target += projection @ origin
    point = np.linalg.solve(system, target)
    offset = point - origins
    along = np.sum(offset * directions, axis=1, keepdims=True)
    distance = np.linalg.norm(offset - along * directions, axis=1)
    return point, distance


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    return center, axes[-1]


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(np.asarray(target) - np.asarray(position))
    up = normalize(up)
    right = normalize(np.cross(forward, up))
    camera_up = np.cross(right, forward)
    rotation_c2w = np.column_stack([right, -camera_up, forward])
    pose = np.eye(4)
    pose[:3, :3] = rotation_c2w.T
    pose[:3, 3] = -rotation_c2w.T @ np.asarray(position, dtype=np.float64)
    return pose


def project(points: np.ndarray, w2c: np.ndarray, intrinsics: np.ndarray):
    points = np.asarray(points, dtype=np.float64)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    pixels = camera @ intrinsics.T
    depth = camera[:, 2]
    uv = pixels[:, :2] / np.maximum(depth[:, None], 1e-9)
    return uv, depth


def principal_direction(points: np.ndarray, up: np.ndarray) -> np.ndarray:
    up = normalize(up)
    centered = np.asarray(points, dtype=np.float64)
    centered = centered - centered.mean(axis=0)
    horizontal = centered - np.outer(centered @ up, up)
    _, _, axes = np.linalg.svd(horizontal, full_matrices=False)
    direction = axes[0]
    direction = direction - up * np.dot(direction, up)
    return normalize(direction)


def unwrap_angles(origins: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Cumulative azimuth in degrees, zero at the first camera, unwrapped in sequence."""
    up = normalize(up)
    offset = np.asarray(origins, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    horizontal = offset - np.outer(offset @ up, up)
    reference = normalize(horizontal[0])
    side = np.cross(up, reference)
    angle = np.arctan2(horizontal @ side, horizontal @ reference)
    return np.degrees(np.unwrap(angle))


def continuous_orbit(anchor_w2c, anchor_k, anchor_angle, center, up, angles) -> dict:
    """Rotate one rigid camera about ``center`` along ``up`` to each target angle."""
    center = np.asarray(center, dtype=np.float64)
    up = normalize(up)
    w2c = np.asarray(anchor_w2c, dtype=np.float64)
    intrinsics = np.asarray(anchor_k, dtype=np.float64)
    camera_to_world = np.linalg.inv(w2c)
    offset = camera_to_world[:3, 3] - center
    targets = []
    for angle in np.asarray(angles, dtype=np.float64):
        spin = _rotation_about(up, -np.deg2rad(angle - anchor_angle))
        pose = np.eye(4)
        pose[:3, :3] = spin @ camera_to_world[:3, :3]
        pose[:3, 3] = center + spin @ offset
        targets.append(np.linalg.inv(pose))
    return dict(target_w2c=targets, target_intrinsics=[intrinsics.copy() for _ in targets])


def _rotation_about(axis: np.ndarray, radians: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    cosine, sine = np.cos(radians), np.sin(radians)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def framing_margin(corners: np.ndarray, w2c: np.ndarray, intrinsics: np.ndarray, size) -> float:
    uv, depth = project(corners, w2c, intrinsics)
    if np.any(depth <= 0.05):
        return -1.0
    width, height = size
    return float(
        min(uv[:, 0].min(), uv[:, 1].min(), width - 1 - uv[:, 0].max(), height - 1 - uv[:, 1].max())
    )
