"""Step 5: build the closed orbit and export the scene the existing scripts read."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np

from gen_pro.geometry import (
    apply_similarity,
    camera_center,
    continuous_orbit,
    framing_margin,
    look_at,
    normalize,
    transform_w2c,
    umeyama,
)


def run(ws, cfg) -> dict:
    overrides = ws.overrides("orbit")
    frame_count = int(overrides.get("frames", cfg.orbit_frames))
    params = dict(
        frames=frame_count,
        radius_scale=float(overrides.get("radius_scale", 1.0)),
        height_offset=float(overrides.get("height_offset", 0.0)),
        margin=cfg.framing_margin_px,
    )
    reference_path = overrides.get("reference_trajectory")
    if reference_path:
        reference_path = Path(reference_path).resolve()
        params = dict(
            reference_trajectory=str(reference_path),
            reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            margin=cfg.framing_margin_px,
        )
    if ws.fresh("orbit", params):
        print("ORBIT_EXISTS", flush=True)
        return ws.record("orbit")["metrics"]
    directory = ws.step_dir("orbit")
    cameras = ws.read_json(ws.step_dir("cameras") / "source_cameras.json")["views"]
    init = ws.read_json(ws.step_dir("cameras") / "initialization.json")
    center = np.asarray(init["center"], dtype=np.float64)
    up = normalize(init["up"])
    positions = np.array([camera_center(row["w2c"]) for row in cameras])
    offset = positions - center
    vertical = offset @ up
    radial = offset - vertical[:, None] * up
    source_radius = float(np.median(np.linalg.norm(radial, axis=1)))
    intrinsics = _median_intrinsics(cameras, cfg.width, cfg.height)
    corners = _corners(init["bounds_world"])
    warnings = []
    reference_fit = None
    frames = [int(row["video_frame"]) for row in ws.read_json(ws.step_dir("sources") / "sources.json")["views"]]
    if reference_path:
        poses, reference_fit = map_reference_targets(ws.read_json(reference_path), frames, positions)
        frame_count = len(poses)
        margins = [framing_margin(corners, pose, intrinsics, (cfg.width, cfg.height)) for pose in poses]
        if min(margins) < cfg.framing_margin_px:
            warnings.append(f"reference orbit leaves only {min(margins):.1f}px around the cloud")
        grown = 0
        final_radius = float(np.median([
            np.linalg.norm((camera_center(pose) - center) - ((camera_center(pose) - center) @ up) * up)
            for pose in poses[:-1]
        ]))
        camera_height = float(np.median([(camera_center(pose) - center) @ up for pose in poses[:-1]]))
        start = normalize(
            (camera_center(poses[0]) - center) - ((camera_center(poses[0]) - center) @ up) * up
        )
    else:
        radius = source_radius * params["radius_scale"]
        camera_height = float(np.median(vertical)) + params["height_offset"]
        start = normalize(radial[0])
        poses, margins, grown = _fit_radius(
            center, up, start, radius, camera_height, intrinsics, corners, frame_count, cfg
        )
        poses[-1] = poses[0]
        final_radius = radius * (cfg.radius_growth ** grown)
        if grown > 3:
            warnings.append("orbit radius grew more than 3 times to frame the cloud")
    extent = _extent(poses[:-1])
    base = _angle_of(start, init)
    angles = base + np.linspace(0, 360, frame_count)
    trajectory = dict(
        source_w2c=[row["w2c"] for row in cameras],
        source_intrinsics=[row["intrinsics"] for row in cameras],
        source_frames=frames,
        target_w2c=[pose.tolist() for pose in poses],
        target_intrinsics=[intrinsics.tolist() for _ in poses],
        target_angles_deg=angles.tolist(),
        unwrapped_angles_deg=angles.tolist(),
        intrinsics_image_size=[cfg.width, cfg.height],
        camera_convention="OpenCV W2C",
    )
    init["scene_extent"] = extent
    ws.write_json(directory / "trajectory.json", trajectory)
    ws.write_json(directory / "initialization.json", init)
    ws.write_json(
        directory / "orbit.json",
        dict(
            radius=final_radius,
            source_radius=source_radius,
            radius_ratio=final_radius / max(source_radius, 1e-6),
            height=camera_height,
            radius_growths=grown,
            margin_min_px=float(min(margins)),
            scene_extent=extent,
            reference_fit=reference_fit,
            warnings=warnings,
        ),
    )
    scene = _export(ws, trajectory, init)
    metrics = dict(
        margin_min_px=float(min(margins)),
        radius_growths=grown,
        radius_ratio=final_radius / max(source_radius, 1e-6),
        scene_extent=extent,
        reference_fit=reference_fit,
        warnings=warnings,
        scene=str(scene),
    )
    ws.complete("orbit", params, ["trajectory.json", "orbit.json", "initialization.json"], metrics)
    print("ORBIT_DONE", metrics, flush=True)
    return metrics


def map_reference_targets(reference: dict, frames: list[int], positions: np.ndarray):
    """Carry another run's target cameras into this frame through the shared source centers."""
    by_frame = {
        int(frame): camera_center(w2c)
        for frame, w2c in zip(reference["source_frames"], reference["source_w2c"])
    }
    missing = [frame for frame in frames if frame not in by_frame]
    if missing:
        raise KeyError(f"Reference trajectory has no source camera for frames {missing}")
    anchors = np.array([by_frame[frame] for frame in frames])
    scale, rotation, translation = umeyama(anchors, positions)
    residual = np.linalg.norm(apply_similarity(anchors, scale, rotation, translation) - positions, axis=1)
    poses = np.stack([transform_w2c(w2c, scale, rotation, translation) for w2c in reference["target_w2c"]])
    spread = np.linalg.norm(positions - positions.mean(axis=0), axis=1).mean()
    fit = dict(
        scale=scale,
        center_residual_ratio=[float(value) for value in residual / max(spread, 1e-6)],
    )
    return poses, fit


def _fit_radius(center, up, start, radius, height, intrinsics, corners, frame_count, cfg):
    margins = [-1.0]
    poses = None
    for grown in range(cfg.framing_attempts + 1):
        anchor = look_at(center + radius * start + height * up, center, up)
        angles = np.linspace(0, 360, frame_count)
        built = continuous_orbit(anchor, intrinsics, 0.0, center, up, angles)
        poses = np.stack(built["target_w2c"])
        margins = [
            framing_margin(corners, pose, intrinsics, (cfg.width, cfg.height)) for pose in poses
        ]
        if min(margins) >= cfg.framing_margin_px:
            return poses, margins, grown
        radius *= cfg.radius_growth
    raise RuntimeError(f"Orbit framing failed; best margin {min(margins):.1f}px")


def _extent(poses) -> float:
    centers = np.array([camera_center(pose) for pose in poses])
    return float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max() * 1.1)


def _angle_of(direction, init) -> float:
    front = np.asarray(init["front"], dtype=np.float64)
    left = np.asarray(init["left"], dtype=np.float64)
    return float(np.degrees(np.arctan2(np.dot(direction, left), np.dot(direction, front))))


def _median_intrinsics(cameras, width, height) -> np.ndarray:
    values = np.asarray([row["intrinsics"] for row in cameras], dtype=np.float64)
    intrinsics = np.eye(3)
    intrinsics[0, 0] = float(np.median(values[:, 0, 0]))
    intrinsics[1, 1] = float(np.median(values[:, 1, 1]))
    intrinsics[0, 2] = (width - 1) / 2
    intrinsics[1, 2] = (height - 1) / 2
    return intrinsics


def _corners(bounds) -> np.ndarray:
    low, high = np.asarray(bounds, dtype=np.float64)
    return np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )


def _export(ws, trajectory, init) -> str:
    scene = ws.root / "scene"
    if scene.exists():
        shutil.rmtree(scene)
    for name in ("source_views", "source_masks"):
        shutil.copytree(ws.step_dir("sources") / name, scene / name)
    shutil.copyfile(ws.step_dir("cameras") / "initial_points.ply", scene / "initial_points.ply")
    ws.write_json(scene / "trajectory.json", trajectory)
    ws.write_json(scene / "initialization.json", init)
    ws.write_json(
        scene / "scene.json",
        dict(scene_extent=init["scene_extent"], surface_points=init["real_surface_points"]),
    )
    return str(scene)
