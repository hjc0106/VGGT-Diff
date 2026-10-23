"""Vehicle-area-normalized exterior alpha and an interior coverage guard."""
import torch
import torch.nn.functional as F


def balanced_silhouette(alpha, mask, uncertainty_pixels=2):
    if alpha.shape != mask.shape or alpha.ndim != 4 or alpha.shape[-1] != 1:
        raise ValueError('Expected matching NHWC alpha/mask arrays with one channel')
    if uncertainty_pixels < 0:
        raise ValueError('uncertainty_pixels must be nonnegative')
    if not bool((mask > .5).any()):
        raise ValueError('A nonempty vehicle mask is required')
    binary = (mask > .5).to(alpha.dtype).permute(0, 3, 1, 2)
    kernel = 2*uncertainty_pixels+1
    outer = F.max_pool2d(binary, kernel, 1, uncertainty_pixels)
    inner = 1-F.max_pool2d(1-binary, kernel, 1, uncertainty_pixels)
    exterior = (1-outer).permute(0, 2, 3, 1)
    interior = inner.permute(0, 2, 3, 1)
    area = binary.sum().clamp_min(1)
    outside_loss = (alpha*exterior).sum()/area
    inside_loss = ((1-alpha)*interior).sum()/interior.sum().clamp_min(1)
    return outside_loss, inside_loss
