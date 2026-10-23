import torch
import pytest
from scripts.car15_object_loss import object_losses


def test_foreground_normalization_ignores_background_area_and_color():
    for size in [16,32]:
        mask=torch.zeros(1,size,size,1);mask[:,6:10,6:10]=1
        gt=torch.full((1,size,size,3),.8);pred=torch.zeros_like(gt);pred[:,6:10,6:10]=.6
        l1,_,_=object_losses(pred,gt,torch.zeros_like(mask),mask)
        assert float(l1)==pytest.approx(.2,abs=1e-6)


def test_outside_opacity_gradient_excludes_object_and_uncertain_boundary():
    mask=torch.zeros(1,24,24,1);mask[:,8:16,8:16]=1
    alpha=torch.ones_like(mask,requires_grad=True)
    _,_,outside=object_losses(torch.zeros(1,24,24,3),torch.zeros(1,24,24,3),alpha,mask)
    outside.backward()
    assert alpha.grad[:,7:17,7:17].sum()==0
    assert alpha.grad[:,0:4].sum()>0


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError):object_losses(torch.zeros(1,8,8,3),torch.zeros(1,8,8,3),torch.zeros(1,8,8,1),torch.zeros(1,8,8,1))
