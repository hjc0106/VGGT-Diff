"""VGGT-Omega cameras and per-pixel points for the six reference views."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def estimate(checkpoint: str, images: list[Image.Image], device: str) -> dict:
    from vggtdiff.omega import OmegaRuntime, _unproject

    runtime = OmegaRuntime(checkpoint, torch.device(device))
    tensors = runtime._preprocess(images)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        layers, patch_start = runtime.model.aggregator(tensors)
    geometry = [None if layer is None else layer.detach() for layer in layers]
    with torch.autocast(device_type="cuda", enabled=False):
        from vggt_omega.utils.pose_enc import encoding_to_camera

        pose = runtime.model.camera_head(geometry, patch_token_start=patch_start)
        depth, confidence = runtime.model.dense_head(
            geometry, images=tensors, patch_token_start=patch_start
        )
        predicted_w2c, predicted_k = encoding_to_camera(pose, tensors.shape[-2:])
        world = _unproject(
            _leading_map(depth).float(),
            predicted_w2c[0].float(),
            predicted_k[0].float(),
        )
    confidence_map = _leading_map(confidence).float()
    if confidence_map.ndim == 4 and confidence_map.shape[-1] == 1:
        confidence_map = confidence_map[..., 0]
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], len(images), axis=0)
    w2c[:, :3, :] = predicted_w2c[0].float().cpu().numpy()
    return dict(
        w2c=w2c,
        intrinsics=predicted_k[0].float().cpu().numpy(),
        world=world.detach().float().cpu().numpy(),
        confidence=confidence_map.detach().float().cpu().numpy(),
        size=tuple(int(value) for value in tensors.shape[-2:]),
    )


def supported_window(width: int, height: int) -> tuple[int, int, int, int]:
    """Centered crop Omega applies before resizing. Returns width, height, left, top."""
    aspect = height / max(width, 1)
    if aspect < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return crop_width, height, left, 0
    if aspect > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return width, crop_height, 0, top
    return width, height, 0, 0


def intrinsics_to_image(omega_k, image_size, omega_size) -> np.ndarray:
    """Map Omega's tensor intrinsics back to the image, including its centered crop."""
    width, height = image_size
    omega_height, omega_width = int(omega_size[0]), int(omega_size[1])
    crop_width, crop_height, left, top = supported_window(width, height)
    transform = np.array(
        [
            [crop_width / omega_width, 0.0, left],
            [0.0, crop_height / omega_height, top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform @ np.asarray(omega_k, dtype=np.float64)


def maps_to_image(maps, image_size, omega_size) -> np.ndarray:
    """Place Omega maps on the image. Pixels outside the centered crop stay NaN."""
    import cv2

    width, height = image_size
    crop_width, crop_height, left, top = supported_window(width, height)
    placed = []
    for view in maps:
        flat = np.asarray(view, dtype=np.float32)
        resized = cv2.resize(flat, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
        if flat.ndim == 3 and resized.ndim == 2:
            resized = resized[..., None]
        canvas = np.full((height, width) + resized.shape[2:], np.nan, dtype=np.float32)
        canvas[top : top + crop_height, left : left + crop_width] = resized
        placed.append(canvas)
    return np.stack(placed)


def _leading_map(value):
    if isinstance(value, (list, tuple)):
        value = value[0]
    while value.ndim > 3 and value.shape[0] == 1 and (value.ndim > 4 or value.shape[-1] > 4):
        value = value[0]
    return value
