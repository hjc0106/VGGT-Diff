import pytest
import torch

from scripts.car15_balanced_silhouette import balanced_silhouette


def test_exterior_cost_is_unchanged_by_empty_background_padding():
    values = []
    for size in [24, 48]:
        mask = torch.zeros(1, size, size, 1)
        mask[:, 8:16, 8:16] = 1
        alpha = mask.clone()
        alpha[:, 2:4, 2:4] = .8
        outside, inside = balanced_silhouette(alpha, mask)
        values.append(float(outside))
        assert float(inside) == 0
    assert values[0] == pytest.approx(values[1], abs=1e-8)
    assert values[0] == pytest.approx(4*.8/64)


def test_coverage_guard_pushes_against_shrinking_the_vehicle():
    mask = torch.zeros(1, 24, 24, 1)
    mask[:, 7:17, 7:17] = 1
    alpha = torch.zeros_like(mask, requires_grad=True)
    outside, inside = balanced_silhouette(alpha, mask)
    assert float(outside) == 0
    assert float(inside) == 1
    inside.backward()
    assert torch.all(alpha.grad[:, 9:15, 9:15] < 0)
    assert alpha.grad[:, :4].sum() == 0


def test_uncertain_edge_band_has_no_silhouette_gradient():
    mask = torch.zeros(1, 24, 24, 1)
    mask[:, 7:17, 7:17] = 1
    alpha = torch.full_like(mask, .5, requires_grad=True)
    exterior, interior = balanced_silhouette(alpha, mask)
    (exterior+interior).backward()
    assert alpha.grad[0, 7, 10, 0] == 0
    assert alpha.grad[0, 5, 10, 0] == 0
    assert alpha.grad[0, 3, 10, 0] > 0
    assert alpha.grad[0, 10, 10, 0] < 0


def test_empty_vehicle_mask_is_rejected():
    mask = torch.zeros(1, 12, 12, 1)
    with pytest.raises(ValueError, match='nonempty'):
        balanced_silhouette(mask, mask)
