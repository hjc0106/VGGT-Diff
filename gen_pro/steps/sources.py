"""Step 3: segment the six references, aim each camera at the car, and resample to 832x480."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from gen_pro.geometry import camera_axes, camera_center, look_at, ray_convergence
from gen_pro.sam3_masks import Segmenter, pick


def run(ws, cfg) -> dict:
    overrides = ws.overrides("sources")
    prompt = overrides.get("prompt") or cfg.prompt
    focal = float(overrides.get("source_focal_px", cfg.source_focal_px))
    params = dict(
        prompt=prompt,
        source_focal_px=focal,
        min_area=cfg.mask_min_area,
        max_area=cfg.mask_max_area,
        candidates=overrides.get("candidates", {}),
        center=overrides.get("center"),
        projection="aim_at_center",
    )
    if ws.fresh("sources", params):
        print("SOURCES_EXISTS", flush=True)
        return ws.record("sources")["metrics"]
    if not cfg.sam3_checkpoint:
        raise ValueError("sources requires cfg.sam3_checkpoint")
    directory = ws.step_dir("sources")
    (directory / "source_views").mkdir(exist_ok=True)
    (directory / "source_masks").mkdir(exist_ok=True)
    video_dir = ws.step_dir("video")
    selection = ws.read_json(ws.step_dir("select") / "selection.json")["selected"]
    sfm = ws.read_json(video_dir / "sfm.json")
    frames = {row["video_frame"]: row for row in ws.read_json(video_dir / "frames.json")["frames"]}
    intrinsics = np.asarray(sfm["intrinsics"], dtype=np.float64)
    distortion = np.asarray(sfm["distortion"], dtype=np.float64)
    segmenter = Segmenter(cfg.sam3_checkpoint, cfg.sam3_root, cfg.device, cfg.sam3_confidence)
    picked = []
    for item in selection:
        source = frames[item["video_frame"]]
        full = np.asarray(Image.open(video_dir / "frames" / source["name"]).convert("RGB"))
        undistorted = cv2.undistort(full, intrinsics, distortion)
        candidates = segmenter.candidates(Image.fromarray(undistorted), prompt)
        _save_candidates(directory / "candidates" / f"{item['index']:02d}", candidates)
        chosen = pick(candidates, cfg.mask_min_area, cfg.mask_max_area, params["candidates"].get(str(item["index"])))
        picked.append((item, np.asarray(source["w2c"], dtype=np.float64), undistorted, chosen))
    rays = [mask_ray(w2c, intrinsics, chosen["mask"]) for _item, w2c, _image, chosen in picked]
    origins = np.array([origin for origin, _direction in rays])
    point, miss = ray_convergence(origins, np.array([direction for _origin, direction in rays]))
    center = np.asarray(params["center"], dtype=np.float64) if params["center"] is not None else point
    rows = []
    reviews = []
    warnings = []
    spread = np.linalg.norm(origins - origins.mean(axis=0), axis=1).mean()
    if params["center"] is None and float(miss.max() / max(spread, 1e-9)) > 0.05:
        warnings.append("mask rays do not meet at one car center")
    up = np.asarray(ws.read_json(video_dir / "axes.json")["up"], dtype=np.float64)
    for item, w2c, undistorted, chosen in picked:
        view, mask, pose, view_k, rotation_deg, reason = aim_view(
            undistorted, chosen["mask"], w2c, intrinsics, center, up, focal, cfg.width, cfg.height
        )
        if reason:
            warnings.append(f"view {item['index']} {reason}")
        stem = f"{item['index']:02d}.png"
        Image.fromarray(view).save(directory / "source_views" / stem)
        Image.fromarray(mask).save(directory / "source_masks" / stem)
        rows.append(
            dict(
                index=item["index"],
                video_frame=item["video_frame"],
                w2c=pose.tolist(),
                intrinsics=view_k.tolist(),
                virtual_rotation_deg=rotation_deg,
                fit_reason=reason,
                candidate=chosen["id"],
                score=chosen["score"],
                area=chosen["area"],
            )
        )
        reviews.append(_overlay(view, mask, f"{item['index']:02d} {item['video_frame']}"))
        print(
            "SOURCE", item["index"], "frame", item["video_frame"],
            "score", round(chosen["score"], 3), "rotation", round(rotation_deg, 1), flush=True,
        )
    _sheet(reviews, directory / "masks_review.jpg")
    ws.write_json(
        directory / "sources.json",
        dict(prompt=prompt, center=center.tolist(), ray_miss=miss.tolist(), views=rows),
    )
    metrics = dict(
        prompt=prompt,
        source_focal_px=focal,
        frames=[row["video_frame"] for row in rows],
        virtual_rotation_deg=[round(row["virtual_rotation_deg"], 2) for row in rows],
        ray_miss_ratio=float(miss.max() / max(spread, 1e-9)),
        warnings=warnings,
    )
    ws.complete("sources", params, ["sources.json", "masks_review.jpg"], metrics)
    print("SOURCES_DONE", metrics, flush=True)
    return metrics


def mask_ray(w2c, intrinsics, mask):
    """World ray from the camera through the center of the mask's bounding box."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Selected mask is empty")
    pixel = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, 1.0])
    direction = w2c[:3, :3].T @ np.linalg.solve(intrinsics, pixel)
    return camera_center(w2c), direction / np.linalg.norm(direction)


def aim_rotation(w2c, center, up) -> np.ndarray:
    """Camera-frame rotation that looks at ``center`` with no roll about the world ``up``."""
    up = np.asarray(up, dtype=np.float64)
    if np.dot(up, camera_axes(w2c)[1]) < 0:
        up = -up
    aimed = look_at(camera_center(w2c), center, up)
    return aimed[:3, :3] @ w2c[:3, :3].T


def aim_view(image, mask, w2c, intrinsics, center, up, focal, width, height, fit_margin=0.05, supersample=2):
    """Rotate the camera about its center toward ``center`` and resample at ``focal``.

    The focal drops only for a view whose car would not fit with ``fit_margin`` on each side.
    """
    rotation = aim_rotation(w2c, center, up)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Selected mask is empty")
    rays = rotation @ np.linalg.solve(intrinsics, np.stack([xs, ys, np.ones_like(xs)]).astype(np.float64))
    if np.any(rays[2] <= 1e-6):
        raise RuntimeError("Car reaches behind the aimed camera")
    tangent_x = np.abs(rays[0] / rays[2]).max()
    tangent_y = np.abs(rays[1] / rays[2]).max()
    fit = min(
        (width - 1) / 2 * (1 - 2 * fit_margin) / max(tangent_x, 1e-9),
        (height - 1) / 2 * (1 - 2 * fit_margin) / max(tangent_y, 1e-9),
    )
    inside = inside_focal(rotation, intrinsics, image.shape[1], image.shape[0], width, height)
    view_focal = max(min(focal, fit), inside)
    reason = None
    if view_focal > fit + 1e-6:
        reason = f"focal {view_focal:.1f} keeps the view inside the frame but crops the car"
    elif view_focal > focal:
        reason = f"focal raised to {view_focal:.1f} to stay inside the frame"
    elif view_focal < focal:
        reason = f"focal lowered to {view_focal:.1f} to fit the car"
    view_k = np.array([[view_focal, 0, (width - 1) / 2], [0, view_focal, (height - 1) / 2], [0, 0, 1]])
    mask_view = _warp(mask.astype(np.uint8) * 255, intrinsics, rotation, view_k, width, height, 1, cv2.INTER_NEAREST)
    view = _warp(image, intrinsics, rotation, view_k, width, height, supersample, cv2.INTER_LINEAR)
    pose = np.eye(4)
    pose[:3, :3] = rotation @ w2c[:3, :3]
    pose[:3, 3] = rotation @ w2c[:3, 3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    return view, mask_view, pose, view_k, angle, reason


def inside_focal(rotation, intrinsics, image_w, image_h, width, height) -> float:
    """Smallest focal whose whole resampled view still comes from inside the source image."""
    half_w, half_h = (width - 1) / 2, (height - 1) / 2
    border = np.array([
        [x, y] for x in (-half_w, 0.0, half_w) for y in (-half_h, 0.0, half_h) if x or y
    ])

    def inside(focal):
        rays = np.column_stack([border / focal, np.ones(len(border))]) @ rotation
        if np.any(rays[:, 2] <= 1e-9):
            return False
        pixels = rays @ intrinsics.T
        uv = pixels[:, :2] / pixels[:, 2:]
        return bool(np.all((uv >= 0) & (uv <= [image_w - 1, image_h - 1])))

    low, high = 1.0, 1e6
    if inside(low):
        return low
    for _step in range(60):
        middle = np.sqrt(low * high)
        if inside(middle):
            high = middle
        else:
            low = middle
    return high


def _warp(image, intrinsics, rotation, view_k, width, height, supersample, interpolation):
    scale = np.diag([supersample, supersample, 1.0])
    big_k = scale @ view_k
    big_k[0, 2] = (width * supersample - 1) / 2
    big_k[1, 2] = (height * supersample - 1) / 2
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsics, None, rotation, big_k, (width * supersample, height * supersample), cv2.CV_32FC1
    )
    warped = cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT)
    if supersample == 1:
        return warped
    return cv2.resize(warped, (width, height), interpolation=cv2.INTER_AREA)


def _save_candidates(directory, candidates) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        Image.fromarray(candidate["mask"].astype(np.uint8) * 255).save(directory / f"{candidate['id']:02d}.png")


def _overlay(view, mask, label) -> Image.Image:
    color = view.copy()
    color[mask > 127] = (0.55 * color[mask > 127] + np.array([40, 180, 80])).astype(np.uint8)
    image = Image.fromarray(color).resize((416, 240))
    canvas = Image.new("RGB", (416, 264), "#181818")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((8, 4), label, fill="white")
    return canvas


def _sheet(tiles, path) -> None:
    columns = 3
    rows = max(1, (len(tiles) + columns - 1) // columns)
    sheet = Image.new("RGB", (416 * columns, 264 * rows), "black")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * 416, (index // 3) * 264))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
