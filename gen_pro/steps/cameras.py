"""Step 4: the E1 initialization cloud from Omega depth on the aimed COLMAP cameras."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gen_pro.geometry import (
    apply_similarity,
    camera_axes,
    camera_center,
    normalize,
    principal_direction,
    umeyama,
)
from gen_pro.omega_dense import estimate, intrinsics_to_image, maps_to_image


def run(ws, cfg) -> dict:
    overrides = ws.overrides("cameras")
    params = dict(
        checkpoint=cfg.omega_checkpoint,
        confidence_quantile=overrides.get("confidence_quantile", cfg.confidence_quantile),
        outlier_drop=overrides.get("outlier_drop", cfg.outlier_drop),
        seeds=cfg.seed_points,
        seed=cfg.cloud_seed,
        max_surface_points=cfg.max_surface_points,
        pose_source="colmap_aimed",
        projection_min_hit=float(overrides.get("projection_min_hit", cfg.projection_min_hit)),
    )
    if ws.fresh("cameras", params):
        print("CAMERAS_EXISTS", flush=True)
        return ws.record("cameras")["metrics"]
    if not cfg.omega_checkpoint:
        raise ValueError("cameras requires cfg.omega_checkpoint")
    directory = ws.step_dir("cameras")
    sources = ws.read_json(ws.step_dir("sources") / "sources.json")["views"]
    images = [
        Image.open(ws.step_dir("sources") / "source_views" / f"{row['index']:02d}.png").convert("RGB")
        for row in sources
    ]
    digest = _source_digest(images)
    cache = directory / "omega.npz"
    omega = _load_cache(cache, digest)
    if omega is None:
        estimated = estimate(cfg.omega_checkpoint, images, cfg.device)
        np.savez_compressed(
            cache,
            w2c=estimated["w2c"],
            intrinsics=estimated["intrinsics"],
            world=estimated["world"].astype(np.float16),
            confidence=estimated["confidence"].astype(np.float16),
            size=np.asarray(estimated["size"]),
            sources=np.array(digest),
        )
        omega = dict(
            w2c=estimated["w2c"],
            intrinsics=estimated["intrinsics"],
            world=estimated["world"],
            confidence=estimated["confidence"],
            size=np.asarray(estimated["size"]),
        )
    image_size = images[0].size
    omega_intrinsics = [
        intrinsics_to_image(omega["intrinsics"][index], image_size, omega["size"])
        for index in range(len(images))
    ]
    aligned = [np.asarray(row["w2c"], dtype=np.float64) for row in sources]
    view_intrinsics = [np.asarray(row["intrinsics"], dtype=np.float64) for row in sources]
    omega_w2c = [np.asarray(pose, dtype=np.float64) for pose in omega["w2c"]]
    colmap_centers = np.array([camera_center(pose) for pose in aligned])
    omega_centers = np.array([camera_center(pose) for pose in omega_w2c])
    scale, rotation, translation = umeyama(omega_centers, colmap_centers)
    rotation_errors = []
    for pose, estimated in zip(aligned, omega_w2c):
        delta = pose[:3, :3] @ (estimated[:3, :3] @ rotation.T).T
        rotation_errors.append(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    mapped = apply_similarity(omega_centers, scale, rotation, translation)
    center_error = np.linalg.norm(mapped - colmap_centers, axis=1)
    spread = np.linalg.norm(colmap_centers - colmap_centers.mean(axis=0), axis=1).mean()
    warnings = []
    if float(center_error.mean() / max(spread, 1e-6)) > cfg.align_warn_translation:
        warnings.append("Omega to COLMAP center residual is large")
    if float(np.max(rotation_errors)) > cfg.align_warn_rotation_deg:
        warnings.append("Omega to COLMAP rotation difference is large")
    focal_ratios = [
        float(view_k[0, 0] / max(float(estimated_k[0, 0]), 1e-6))
        for view_k, estimated_k in zip(view_intrinsics, omega_intrinsics)
    ]
    masks = [
        np.asarray(Image.open(ws.step_dir("sources") / "source_masks" / f"{row['index']:02d}.png").convert("L")) > 127
        for row in sources
    ]
    world = depth_to_world(
        maps_to_image(omega["world"], image_size, omega["size"]), omega_w2c, scale, aligned, view_intrinsics
    )
    up = normalize(ws.read_json(ws.step_dir("video") / "axes.json")["up"])
    if np.dot(up, np.mean([camera_axes(pose)[1] for pose in aligned], axis=0)) < 0:
        up = -up
    surface, colors, frame = _surface(omega, world, masks, images, up, params, cfg.voxel_divisions)
    if len(surface) < 1000:
        raise RuntimeError(f"Surface cloud has {len(surface)} points, need at least 1000")
    hits = [
        projection_hit(surface, pose, intrinsics, mask)
        for pose, intrinsics, mask in zip(aligned, view_intrinsics, masks)
    ]
    seeds, seed_colors = _seeds(surface, colors, frame, cfg)
    cloud = np.vstack([surface, seeds])
    cloud_colors = np.vstack([colors, seed_colors])
    _write_ply(directory / "initial_points.ply", cloud, cloud_colors)
    bounds = [surface.min(0).tolist(), surface.max(0).tolist()]
    ws.write_json(
        directory / "source_cameras.json",
        dict(views=[dict(index=row["index"], w2c=pose.tolist(), intrinsics=k.tolist())
                     for row, pose, k in zip(sources, aligned, view_intrinsics)]),
    )
    ws.write_json(
        directory / "initialization.json",
        dict(
            real_surface_points=int(len(surface)),
            volume_seed_points=int(len(seeds)),
            seed_opacity=cfg.seed_opacity,
            bounds_world=bounds,
            center=frame["center"].tolist(),
            front=frame["front"].tolist(),
            left=frame["left"].tolist(),
            up=frame["up"].tolist(),
        ),
    )
    ws.write_json(
        directory / "align.json",
        dict(
            scale=scale,
            center_residual=float(center_error.mean()),
            center_residual_ratio=float(center_error.mean() / max(spread, 1e-6)),
            rotation_deg=[float(value) for value in rotation_errors],
            focal_ratio=focal_ratios,
            projection_hit=[float(value) for value in hits],
            warnings=warnings,
        ),
    )
    outputs = ["source_cameras.json", "initialization.json", "initial_points.ply", "align.json"]
    aimed_at = np.asarray(ws.read_json(ws.step_dir("sources") / "sources.json")["center"], dtype=np.float64)
    distance = np.linalg.norm(colmap_centers - frame["center"], axis=1).mean()
    metrics = dict(
        cloud_center=frame["center"].tolist(),
        aim_shift_ratio=float(np.linalg.norm(frame["center"] - aimed_at) / max(distance, 1e-9)),
        surface_points=int(len(surface)),
        center_residual_ratio=float(center_error.mean() / max(spread, 1e-6)),
        max_rotation_deg=float(np.max(rotation_errors)),
        projection_hit=[float(value) for value in hits],
        warnings=warnings,
    )
    minimum = params["projection_min_hit"]
    if float(min(hits)) < minimum:
        ws.fail("cameras", params, outputs, metrics)
        listed = ", ".join(f"view {index} {hit:.2f}" for index, hit in enumerate(hits) if hit < minimum)
        raise RuntimeError(
            f"Projected surface misses the source masks ({listed}); need at least {minimum:.2f} in each view"
        )
    ws.complete("cameras", params, outputs, metrics)
    print("CAMERAS_DONE", metrics, flush=True)
    return metrics


def projection_hit(points, pose, intrinsics, mask) -> float:
    """Fraction of in-image projections that land inside the mask."""
    points = np.asarray(points, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    height, width = mask.shape
    camera = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    front = camera[:, 2] > 1e-3
    if not np.any(front):
        return 0.0
    pixels = (intrinsics @ camera[front].T).T
    uv = pixels[:, :2] / pixels[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[inside]
    if len(uv) == 0:
        return 0.0
    coords = uv.astype(int)
    return float(mask[coords[:, 1], coords[:, 0]].mean())


def depth_to_world(omega_world, omega_w2c, scale, poses, intrinsics) -> np.ndarray:
    """Keep Omega's per-pixel depth, scaled to COLMAP units, and cast it along the known rays."""
    height, width = omega_world.shape[1:3]
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    world = np.empty_like(omega_world, dtype=np.float64)
    for index, (points, estimated, pose, view_k) in enumerate(zip(omega_world, omega_w2c, poses, intrinsics)):
        depth = scale * (points.astype(np.float64) @ estimated[:3, :3].T + estimated[:3, 3])[..., 2]
        camera = (pixels @ np.linalg.inv(view_k).T) * depth[..., None]
        world[index] = (camera - pose[:3, 3]) @ pose[:3, :3]
    return world


def _surface(omega, world, masks, images, up, params, voxel_divisions):
    image_size = (masks[0].shape[1], masks[0].shape[0])
    confidence_maps = np.asarray(omega["confidence"])
    while confidence_maps.ndim > 3 and confidence_maps.shape[0] == 1 and (
        confidence_maps.ndim > 4 or confidence_maps.shape[-1] > 4
    ):
        confidence_maps = confidence_maps[0]
    if confidence_maps.ndim == 4 and confidence_maps.shape[-1] == 1:
        confidence_maps = confidence_maps[..., 0]
    confidence = maps_to_image(confidence_maps, image_size, omega["size"])
    points, colors, weights = [], [], []
    for index, (mask, image) in enumerate(zip(masks, images)):
        valid = mask & np.isfinite(world[index]).all(-1) & np.isfinite(confidence[index])
        if valid.sum() < 100:
            continue
        threshold = np.quantile(confidence[index][valid], params["confidence_quantile"])
        valid &= confidence[index] >= threshold
        points.append(world[index][valid])
        colors.append(np.asarray(image)[valid])
        weights.append(confidence[index][valid])
    xyz = np.concatenate(points)
    rgb = np.concatenate(colors)
    front = principal_direction(xyz, up)
    left = np.cross(up, front)
    left = left / np.linalg.norm(left)
    axes = np.column_stack([front, left, up])
    local = xyz @ axes
    lo, hi = np.quantile(local, [0.025, 0.975], axis=0)
    center = axes @ ((lo + hi) / 2)
    local = (xyz - center) @ axes
    lo, hi = np.quantile(local, [0.01, 0.99], axis=0)
    extent = np.maximum(hi - lo, 1e-4)
    keep = ((local > lo - 0.15 * extent) & (local < hi + 0.15 * extent)).all(1)
    xyz, rgb, local = xyz[keep], rgb[keep], local[keep]
    cell = float(extent.max()) / voxel_divisions
    voxels = np.floor(xyz / cell).astype(np.int64)
    _, unique = np.unique(voxels, axis=0, return_index=True)
    xyz, rgb = xyz[unique], rgb[unique]
    distance = cKDTree(xyz).query(xyz, k=min(8, len(xyz)))[0][:, -1]
    keep = distance <= np.quantile(distance, 1 - params["outlier_drop"])
    xyz, rgb = xyz[keep], rgb[keep]
    cap = params["max_surface_points"]
    if len(xyz) > cap:
        rng = np.random.default_rng(params["seed"])
        choice = rng.choice(len(xyz), cap, replace=False)
        xyz, rgb = xyz[choice], rgb[choice]
    local = (xyz - center) @ axes
    lo, hi = np.quantile(local, [0.01, 0.99], axis=0)
    return xyz, rgb, dict(center=center, front=front, left=left, up=up, lo=lo, hi=hi, axes=axes)


def _seeds(surface, colors, frame, cfg):
    rng = np.random.default_rng(cfg.cloud_seed)
    extent = np.maximum(frame["hi"] - frame["lo"], 1e-4)
    local = rng.uniform(frame["lo"] - 0.1 * extent, frame["hi"] + 0.1 * extent, (cfg.seed_points, 3))
    seeds = local @ frame["axes"].T + frame["center"]
    nearest = cKDTree(surface).query(seeds)[1]
    return seeds, colors[nearest]


def _source_digest(images) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.tobytes())
    return digest.hexdigest()


def _load_cache(path, digest: str):
    if not path.exists():
        return None
    with np.load(path) as cached:
        if "sources" not in cached.files or str(cached["sources"].item()) != digest:
            return None
        return {key: cached[key] for key in cached.files if key != "sources"}


def _write_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement

    array = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    array["x"], array["y"], array["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    array["red"], array["green"], array["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(array, "vertex")]).write(str(path))
