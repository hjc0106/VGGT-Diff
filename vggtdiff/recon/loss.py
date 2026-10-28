"""Foreground-normalized RGB and outside-opacity supervision for E1."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def object_losses(pred, gt, alpha, mask, ssim_fn=None, dilate: int = 5):
    """Match the E1 object loss: foreground L1, crop SSIM, and outside opacity."""
    if mask is None or not bool((mask > 0.5).any()):
        raise ValueError("Object training requires a nonempty mask")
    l1 = ((pred - gt).abs() * mask).sum() / (3 * mask.sum().clamp_min(1))
    structural = pred.new_zeros(())
    if ssim_fn is not None:
        ys, xs = torch.where(mask[0, :, :, 0] > 0.5)
        y0 = max(0, int(ys.min()) - 4)
        y1 = min(mask.shape[1], int(ys.max()) + 5)
        x0 = max(0, int(xs.min()) - 4)
        x1 = min(mask.shape[2], int(xs.max()) + 5)

        def crop(image):
            return (image * mask)[:, y0:y1, x0:x1].permute(0, 3, 1, 2)

        structural = 1 - ssim_fn(crop(pred), crop(gt))
    if dilate < 1 or dilate % 2 == 0:
        raise ValueError("dilate must be a positive odd size")
    expanded = F.max_pool2d(mask.permute(0, 3, 1, 2), dilate, 1, dilate // 2).permute(0, 2, 3, 1)
    outside = 1 - expanded
    opacity = (alpha * outside).sum() / outside.sum().clamp_min(1)
    return l1, structural, opacity
