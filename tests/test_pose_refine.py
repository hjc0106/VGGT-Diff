"""Bounded per-view camera corrections used by --pose-refine."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggtdiff.recon.reconstruction import POSE_REFINE, PoseCorrection


def correction(count=8, translation_max=0.05):
    return PoseCorrection(count, math.radians(POSE_REFINE["rotation_max_deg"]), translation_max, "cpu")


def test_zero_correction_is_the_identity():
    delta = correction().delta(3)
    assert torch.allclose(delta, torch.eye(4), atol=1e-6)


def test_correction_stays_inside_its_bounds():
    limit = 0.05
    model = correction(translation_max=limit)
    with torch.no_grad():
        model.raw.fill_(100)
    omega, translation = model.motion(1)
    assert torch.linalg.vector_norm(omega) <= math.radians(1.0) + 1e-5
    assert torch.linalg.vector_norm(translation) <= limit + 1e-5


def test_the_anchor_view_is_not_updated():
    model = correction()
    model.raw.grad = torch.ones_like(model.raw)
    before = model.raw.detach().clone()
    model.step(0, 1e-2)
    assert torch.equal(model.raw.detach(), before)
    assert model.raw.grad is None


def test_only_the_current_view_moves():
    model = correction()
    model.raw.grad = torch.ones_like(model.raw)
    model.step(5, 1e-2)
    assert model.raw[5].abs().sum() > 0
    assert torch.equal(model.raw[4].detach(), torch.zeros(6))
    assert torch.equal(model.m[4], torch.zeros(6))
    assert int(model.steps[5]) == 1
    assert int(model.steps[4]) == 0


def test_correction_recovers_a_known_camera_offset():
    if not torch.cuda.is_available():
        return
    from gsplat import rasterization

    device = "cuda"
    view = 1
    count = 200
    torch.manual_seed(0)
    means = torch.randn(count, 3, device=device) * 0.2
    means[:, 2] += 3.0
    quats = torch.nn.functional.normalize(torch.randn(count, 4, device=device), dim=-1)
    scales = torch.full((count, 3), 0.04, device=device)
    opacity = torch.full((count,), 0.8, device=device)
    colors = torch.rand(count, 1, 3, device=device)
    intrinsics = torch.tensor([[220.0, 0.0, 64.0], [0.0, 220.0, 48.0], [0.0, 0.0, 1.0]], device=device)

    def render(camera_to_world):
        image, _alpha, _info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacity,
            colors=colors,
            viewmats=torch.linalg.inv(camera_to_world)[None],
            Ks=intrinsics[None],
            width=128,
            height=96,
            sh_degree=0,
            packed=False,
        )
        return image

    with torch.no_grad():
        offset = PoseCorrection(2, math.radians(1), 0.05, device)
        offset.raw[view, 0] = 0.4
        offset.raw[view, 4] = 0.3
        target_camera = offset.delta(view).detach()
    target = render(target_camera).detach()
    model = PoseCorrection(2, math.radians(2), 0.1, device)
    for _step in range(500):
        image = render(model.delta(view))
        ((image - target) ** 2).mean().backward()
        model.step(view, 5e-2)
    recovered = model.delta(view).detach()
    relative = torch.linalg.inv(recovered) @ target_camera
    cosine = ((relative[:3, :3].trace() - 1) / 2).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))
    shift = torch.linalg.vector_norm(relative[:3, 3])
    assert float(angle) < 0.05, float(angle)
    assert float(shift) < 1e-3, float(shift)
    assert torch.equal(model.raw[0].detach(), torch.zeros(6, device=device))


if __name__ == "__main__":
    test_zero_correction_is_the_identity()
    test_correction_stays_inside_its_bounds()
    test_the_anchor_view_is_not_updated()
    test_only_the_current_view_moves()
    test_correction_recovers_a_known_camera_offset()
    print("OK")
