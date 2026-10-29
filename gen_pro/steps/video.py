"""Step 1: sample sharp frames, run SfM, and estimate the object center and up axis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gen_pro.colmap_io import register_new_images, run_colmap
from gen_pro.geometry import camera_axes, camera_center, fit_plane, ray_convergence, unwrap_angles


def run(ws, cfg) -> dict:
    video = Path(cfg.video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    overrides = ws.overrides("video")
    params = dict(
        video=_identity(video),
        candidates=cfg.candidate_count,
        window=cfg.sharpness_window,
        max_image_size=cfg.sfm_max_image_size,
        overlap=cfg.sfm_overlap,
        center=overrides.get("center"),
        up=overrides.get("up"),
    )
    if ws.fresh("video", params):
        print("VIDEO_EXISTS", ws.step_dir("video"), flush=True)
        return ws.record("video")["metrics"]
    directory = ws.step_dir("video")
    saved = directory / "frames.json"
    if saved.exists() and ws.read_json(saved).get("identity") == params["video"]:
        frames = ws.read_json(saved)["frames"]
        print("FRAMES_EXIST", len(frames), flush=True)
    else:
        frames = _extract(video, directory / "frames", cfg)
        ws.write_json(directory / "frames.json", dict(identity=params["video"], frames=frames))
    model = run_colmap(directory / "frames", directory, cfg)
    by_name = {image["name"]: image for image in model["images"]}
    camera = next(iter(model["cameras"].values()))
    if camera["model"] != "OPENCV":
        raise RuntimeError(f"Expected an OPENCV camera, got {camera['model']}")
    fx, fy, cx, cy, k1, k2, p1, p2 = camera["params"]
    intrinsics = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    registered = []
    for frame in frames:
        image = by_name.get(frame["name"])
        frame["registered"] = image is not None
        if image is not None:
            frame["w2c"] = image["w2c"]
            registered.append(frame)
    ratio = len(registered) / max(len(frames), 1)
    if ratio < cfg.sfm_min_register_ratio:
        raise RuntimeError(f"SfM registered {ratio:.0%} of the candidate frames")
    axes = _axes(registered, overrides)
    for frame in frames:
        frame["azimuth_deg"] = None
    lookup = {row["name"]: row["azimuth_deg"] for row in axes["frames"]}
    for frame in frames:
        frame["azimuth_deg"] = lookup.get(frame["name"])
    ws.write_json(
        directory / "frames.json",
        dict(identity=params["video"], fps=frames[0]["fps"], frames=frames),
    )
    ws.write_json(
        directory / "sfm.json",
        dict(
            intrinsics=intrinsics,
            distortion=[k1, k2, p1, p2],
            width=camera["width"],
            height=camera["height"],
            points=model["points"],
            registered=len(registered),
            candidates=len(frames),
        ),
    )
    ws.write_json(directory / "axes.json", axes["summary"])
    warnings = []
    if axes["summary"]["azimuth_span_deg"] < 90:
        warnings.append("camera azimuth covers less than 90 degrees")
    if axes["summary"]["convergence_residual_ratio"] > 0.15:
        warnings.append("camera rays do not meet tightly")
    metrics = dict(
        candidates=len(frames),
        registered=len(registered),
        register_ratio=ratio,
        points=model["points"],
        azimuth_span_deg=axes["summary"]["azimuth_span_deg"],
        warnings=warnings,
    )
    ws.complete("video", params, ["frames.json", "sfm.json", "axes.json"], metrics)
    print("VIDEO_DONE", metrics, flush=True)
    return metrics


def add_manual_frames(ws, cfg, frame_ids: list[int]) -> None:
    """Register exact video frames that the uniform sample did not keep."""
    directory = ws.step_dir("video")
    payload = ws.read_json(directory / "frames.json")
    have = {int(row["video_frame"]) for row in payload["frames"]}
    missing = [int(frame) for frame in frame_ids if int(frame) not in have]
    if not missing:
        return
    rows = _extract_exact(Path(cfg.video), directory / "frames", payload["frames"], missing, cfg)
    model = register_new_images(directory / "frames", directory, [row["name"] for row in rows], cfg)
    by_name = {image["name"]: image for image in model["images"]}
    failed = [row["name"] for row in rows if row["name"] not in by_name]
    if failed:
        raise RuntimeError(f"COLMAP did not register {failed}")
    payload["frames"].extend(rows)
    for frame in payload["frames"]:
        image = by_name.get(frame["name"])
        if image is not None:
            frame["w2c"] = image["w2c"]
            frame["registered"] = True
    registered = [frame for frame in payload["frames"] if frame.get("registered")]
    axes = ws.read_json(directory / "axes.json")
    ordered = sorted(registered, key=lambda row: row["video_frame"])
    origins = np.array([camera_center(row["w2c"]) for row in ordered])
    angles = unwrap_angles(origins, np.asarray(axes["center"]), np.asarray(axes["up"]))
    for frame, angle in zip(ordered, angles):
        frame["azimuth_deg"] = float(angle)
    ws.write_json(directory / "frames.json", payload)
    sfm = ws.read_json(directory / "sfm.json")
    sfm["registered"] = len(registered)
    sfm["candidates"] = len(payload["frames"])
    sfm["points"] = model["points"]
    ws.write_json(directory / "sfm.json", sfm)
    print("FRAMES_ADDED", [row["video_frame"] for row in rows], flush=True)


def _extract_exact(video: Path, dest: Path, existing: list[dict], frame_ids: list[int], cfg) -> list[dict]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video), ctx=cpu(0))
    count = len(reader)
    fps = float(existing[0]["fps"])
    rows = []
    for frame_id in frame_ids:
        if frame_id < 0 or frame_id >= count:
            raise IndexError(f"Video frame {frame_id} is outside 0..{count - 1}")
        earlier = [row for row in existing if row["video_frame"] < frame_id]
        slot = earlier[-1]["slot"] if earlier else 0
        frame = reader.get_batch([frame_id]).asnumpy()[0]
        name = f"{slot:04d}_{frame_id:06d}.jpg"
        Image.fromarray(frame).save(dest / name, quality=cfg.jpeg_quality)
        rows.append(
            dict(
                slot=int(slot),
                video_frame=int(frame_id),
                time_sec=frame_id / fps,
                sharpness=float(_sharpness(frame)),
                name=name,
                fps=fps,
                registered=False,
                azimuth_deg=None,
            )
        )
        print("FRAME_EXACT", name, flush=True)
    return rows


def _extract(video: Path, dest: Path, cfg) -> list[dict]:
    from decord import VideoReader, cpu

    dest.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(str(video), ctx=cpu(0))
    count = len(reader)
    fps = float(reader.get_avg_fps())
    slots = np.unique(np.linspace(0, count - 1, cfg.candidate_count).astype(int))
    rows = []
    for slot, position in enumerate(slots):
        start = max(0, int(position) - cfg.sharpness_window)
        stop = min(count, int(position) + cfg.sharpness_window + 1)
        batch = reader.get_batch(list(range(start, stop))).asnumpy()
        scores = [_sharpness(frame) for frame in batch]
        offset = int(np.argmax(scores))
        index = start + offset
        name = f"{slot:04d}_{index:06d}.jpg"
        Image.fromarray(batch[offset]).save(dest / name, quality=cfg.jpeg_quality)
        rows.append(
            dict(
                slot=int(slot),
                video_frame=int(index),
                time_sec=index / fps,
                sharpness=float(scores[offset]),
                name=name,
                fps=fps,
            )
        )
        print("FRAME", name, "sharpness", round(scores[offset], 1), flush=True)
    return rows


def _axes(registered: list[dict], overrides: dict) -> dict:
    poses = np.asarray([row["w2c"] for row in registered], dtype=np.float64)
    origins = np.array([camera_center(pose) for pose in poses])
    forwards = np.array([camera_axes(pose)[0] for pose in poses])
    ups = np.array([camera_axes(pose)[1] for pose in poses])
    center, residual = ray_convergence(origins, forwards)
    _, normal = fit_plane(origins)
    if np.dot(normal, ups.mean(axis=0)) < 0:
        normal = -normal
    if overrides.get("center") is not None:
        center = np.asarray(overrides["center"], dtype=np.float64)
    if overrides.get("up") is not None:
        normal = np.asarray(overrides["up"], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
    angles = unwrap_angles(origins, center, normal)
    distance = np.linalg.norm(origins - center, axis=1)
    for row, angle in zip(registered, angles):
        row["azimuth_deg"] = float(angle)
    summary = dict(
        center=center.tolist(),
        up=normal.tolist(),
        convergence_residual_m=float(np.median(residual)),
        convergence_residual_ratio=float(np.median(residual) / max(np.median(distance), 1e-6)),
        azimuth_span_deg=float(angles.max() - angles.min()),
        frames=[dict(name=row["name"], azimuth_deg=row["azimuth_deg"]) for row in registered],
    )
    return dict(summary=summary, frames=summary["frames"])


def _sharpness(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (320, max(1, int(320 * height / width))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _identity(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
