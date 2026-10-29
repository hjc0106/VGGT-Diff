"""Step 2: choose six reference frames spread across the observed azimuth."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def choose_references(rows: list[dict], count: int, quantile: float, manual_frames: list[int]):
    registered = [row for row in rows if row.get("registered")]
    if len(registered) < count:
        raise RuntimeError(f"Need {count} registered frames, found {len(registered)}")
    threshold = float(np.quantile([row["sharpness"] for row in registered], quantile))
    pool = [row for row in registered if row["sharpness"] >= threshold] or registered
    if manual_frames:
        by_frame = {int(row["video_frame"]): row for row in registered}
        missing = [frame for frame in manual_frames if frame not in by_frame]
        if missing:
            raise KeyError(f"Manual frames are not registered: {missing}")
        # Keep the given order: view 0 is the orbit start.
        chosen = [by_frame[int(frame)] for frame in manual_frames]
        reason = "manual"
    else:
        if len(pool) < count:
            pool = registered
        chosen = _spread(pool, count)
        reason = "azimuth"
        chosen = sorted(chosen, key=lambda row: row["azimuth_deg"])
    return chosen, reason, threshold


def run(ws, cfg) -> dict:
    overrides = ws.overrides("select")
    manual = list(overrides.get("frames") or cfg.manual_frames)
    params = dict(count=cfg.reference_count, quantile=cfg.sharpness_quantile, frames=manual)
    if ws.fresh("select", params):
        print("SELECT_EXISTS", flush=True)
        return ws.record("select")["metrics"]
    if manual:
        from gen_pro.steps.video import add_manual_frames

        add_manual_frames(ws, cfg, [int(frame) for frame in manual])
    video = ws.read_json(ws.step_dir("video") / "frames.json")
    chosen, reason, threshold = choose_references(
        video["frames"], cfg.reference_count, cfg.sharpness_quantile, manual
    )
    for index, row in enumerate(chosen):
        row["index"] = index
    directory = ws.step_dir("select")
    _contact(ws.step_dir("video") / "frames", video["frames"], chosen, directory / "selection.jpg")
    payload = dict(
        reason=reason,
        sharpness_threshold=threshold,
        selected=[
            dict(
                index=row["index"],
                video_frame=row["video_frame"],
                name=row["name"],
                azimuth_deg=row["azimuth_deg"],
                sharpness=row["sharpness"],
            )
            for row in chosen
        ],
    )
    ws.write_json(directory / "selection.json", payload)
    metrics = dict(
        reason=reason,
        frames=[row["video_frame"] for row in chosen],
        azimuth=[round(row["azimuth_deg"], 2) for row in chosen],
    )
    ws.complete("select", params, ["selection.json", "selection.jpg"], metrics)
    print("SELECT_DONE", metrics, flush=True)
    return metrics


def _spread(pool: list[dict], count: int) -> list[dict]:
    ordered = sorted(pool, key=lambda row: row["azimuth_deg"])
    angles = np.array([row["azimuth_deg"] for row in ordered], dtype=float)
    targets = np.linspace(angles.min(), angles.max(), count)
    used = set()
    chosen = []
    for target in targets:
        order = np.argsort(np.abs(angles - target))
        for index in order:
            if int(index) not in used:
                used.add(int(index))
                chosen.append(ordered[int(index)])
                break
    return chosen


def _contact(frame_dir, rows, chosen, path) -> None:
    picked = {row["name"] for row in chosen}
    tiles = []
    for row in rows:
        if row["name"] not in picked and row["slot"] % 4 != 0:
            continue
        image = Image.open(frame_dir / row["name"]).convert("RGB").resize((320, 180))
        canvas = Image.new("RGB", (320, 200), "#16351f" if row["name"] in picked else "#202020")
        canvas.paste(image, (0, 20))
        angle = row.get("azimuth_deg")
        label = f"{row['video_frame']}" if angle is None else f"{row['video_frame']}  {angle:.0f}deg"
        ImageDraw.Draw(canvas).text((6, 2), label, fill="white")
        tiles.append(canvas)
    columns = 6
    rows_n = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (320 * columns, 200 * rows_n), "black")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 320, (index // columns) * 200))
    sheet.save(path, quality=90)
