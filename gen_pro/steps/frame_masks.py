"""Step 6: segment the generated frames after Trajectory inference."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from gen_pro.sam3_masks import Segmenter, pick
from gen_pro.steps.sources import _overlay, _save_candidates, _sheet


def area_jumps(areas: list[float], index: int) -> bool:
    """True when this frame's mask area is more than 3x, or less than 1/3, of its neighbors."""
    neighbors = [
        areas[other]
        for other in (index - 1, index + 1)
        if 0 <= other < len(areas) and areas[other] > 0
    ]
    if not neighbors or areas[index] <= 0:
        return False
    reference = float(np.median(neighbors))
    return areas[index] > 3 * reference or areas[index] < reference / 3


def run(ws, cfg) -> dict:
    frames_dir = Path(cfg.generated_frames)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    overrides = ws.overrides("frame_masks")
    sources = ws.record("sources")
    prompt = overrides.get("prompt") or (sources or {}).get("params", {}).get("prompt") or cfg.prompt
    chosen_ids = {str(key): value for key, value in overrides.get("candidates", {}).items()}
    params = dict(prompt=prompt, frames=str(frames_dir.resolve()), candidates=chosen_ids)
    if ws.fresh("frame_masks", params):
        print("FRAME_MASKS_EXIST", flush=True)
        return ws.record("frame_masks")["metrics"]
    if not cfg.sam3_checkpoint:
        raise ValueError("frame masks require cfg.sam3_checkpoint")
    directory = ws.step_dir("frame_masks")
    mask_dir = directory / "masks"
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir()
    segmenter = Segmenter(cfg.sam3_checkpoint, cfg.sam3_root, cfg.device, cfg.sam3_confidence)
    paths = sorted(frames_dir.glob("*.png"))
    rows = []
    review_tiles = []
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        found = segmenter.candidates(image, prompt)
        _save_candidates(directory / "candidates" / f"{index:02d}", found)
        override = chosen_ids.get(str(index))
        chosen = None
        reason = None
        if override is not None:
            chosen = pick(found, cfg.mask_min_area, cfg.mask_max_area, int(override))
        else:
            try:
                chosen = pick(found, cfg.mask_min_area, cfg.mask_max_area, require_range=True)
            except RuntimeError:
                reason = "no candidate in range"
                low = segmenter.candidates(image, prompt, confidence=0.01)
                _save_candidates(directory / "candidates_low" / f"{index:02d}", low)
                if low:
                    best = max(low, key=lambda item: item["score"])
                    review_tiles.append(_overlay(
                        np.asarray(image), best["mask"].astype(np.uint8) * 255, f"{index:02d} {reason}"
                    ))
        if chosen is None:
            rows.append(dict(index=index, candidate=None, score=0.0, area=0, fraction=0.0, review=reason))
            print("FRAME_MASK", index, reason, flush=True)
            continue
        Image.fromarray(chosen["mask"].astype(np.uint8) * 255).save(mask_dir / f"{index:02d}.png")
        rows.append(dict(
            index=index,
            candidate=chosen["id"],
            score=chosen["score"],
            area=chosen["area"],
            fraction=chosen["fraction"],
            box=chosen["box"],
        ))
        print("FRAME_MASK", index, "area", chosen["area"], flush=True)
    areas = [row["area"] for row in rows]
    for row in rows:
        if row.get("review") or str(row["index"]) in chosen_ids:
            continue
        if area_jumps(areas, row["index"]):
            row["review"] = "area jumps relative to neighboring frames"
            frame = np.asarray(Image.open(paths[row["index"]]).convert("RGB"))
            mask = np.asarray(Image.open(mask_dir / f"{row['index']:02d}.png"))
            review_tiles.append(_overlay(frame, mask, f"{row['index']:02d} {row['review']}"))
            print("FRAME_MASK", row["index"], row["review"], flush=True)
    pending = [row for row in rows if row.get("review")]
    ws.write_json(directory / "frame_masks.json", dict(prompt=prompt, frames=rows, needs_review=[row["index"] for row in pending]))
    if review_tiles:
        _sheet(review_tiles, directory / "needs_review.jpg")
    metrics = dict(frames=len(rows), prompt=prompt, needs_review=[row["index"] for row in pending])
    if pending:
        ws.fail("frame_masks", params, ["frame_masks.json"], metrics)
        listed = ", ".join(str(row["index"]) for row in pending)
        raise RuntimeError(
            f"{len(pending)} frames need a mask choice in {directory / 'overrides.json'}: {listed}"
        )
    ws.complete("frame_masks", params, ["frame_masks.json", "masks"], metrics)
    print("FRAME_MASKS_DONE", metrics, flush=True)
    return metrics
