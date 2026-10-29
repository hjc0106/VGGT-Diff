"""SAM3 text-prompt masks, without the WorldAct wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Segmenter:
    def __init__(self, checkpoint: str, root: str | None, device: str, confidence: float):
        if root:
            sys.path.insert(0, str(Path(root).resolve()))
        self._patch_dtype()
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=False,
            device=device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(model, device=device, confidence_threshold=confidence)

    def candidates(self, image: Image.Image, prompt: str, confidence: float | None = None) -> list[dict]:
        previous = self.processor.confidence_threshold
        if confidence is not None:
            self.processor.confidence_threshold = confidence
        try:
            return self._candidates(image, prompt)
        finally:
            self.processor.confidence_threshold = previous

    def _candidates(self, image: Image.Image, prompt: str) -> list[dict]:
        state = self.processor.set_image(image.convert("RGB"))
        output = self.processor.set_text_prompt(prompt=prompt, state=state)
        masks = output.get("masks")
        if masks is None or masks.numel() == 0:
            return []
        masks = masks.detach().float()
        if masks.ndim == 4:
            masks = masks[:, 0]
        scores = output.get("scores")
        if scores is None:
            scores = torch.ones(masks.shape[0])
        found = []
        for index in range(masks.shape[0]):
            mask = masks[index].cpu().numpy() > 0.5
            area = int(mask.sum())
            if area == 0:
                continue
            ys, xs = np.where(mask)
            found.append(
                dict(
                    id=len(found),
                    score=float(scores[index].detach().cpu()),
                    area=area,
                    fraction=area / float(mask.size),
                    box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    mask=mask,
                )
            )
        return found

    @staticmethod
    def _patch_dtype() -> None:
        try:
            import sam3.model.vitdet as vitdet
        except ImportError:
            return
        if not hasattr(vitdet, "addmm_act"):
            return
        if getattr(vitdet.addmm_act, "_gen_pro_patch", False):
            return
        original = vitdet.addmm_act

        def patched(activation, linear, mat1):
            return original(activation, linear, mat1).to(dtype=linear.weight.dtype)

        patched._gen_pro_patch = True
        vitdet.addmm_act = patched


def pick(
    candidates: list[dict],
    min_area: float,
    max_area: float,
    candidate_id: int | None = None,
    require_range: bool = False,
):
    if not candidates:
        raise RuntimeError("SAM3 returned no mask")
    if candidate_id is not None:
        matches = [item for item in candidates if item["id"] == int(candidate_id)]
        if not matches:
            raise KeyError(f"Mask candidate {candidate_id} does not exist")
        return matches[0]
    eligible = [item for item in candidates if min_area < item["fraction"] < max_area]
    if not eligible:
        if require_range:
            raise RuntimeError("SAM3 returned no mask in the area range")
        eligible = candidates
    return max(eligible, key=lambda item: item["score"])
