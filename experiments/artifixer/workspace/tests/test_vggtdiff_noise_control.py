import pytest
import torch
from scripts.vggtdiff_noise_control import select_noise, noise_digest


def test_public_camera_noise_is_identical_after_subset_expansion_and_reordering():
    bank=torch.arange(2*14*2*3,dtype=torch.float32).reshape(1,2,14,2,3).to(torch.bfloat16)
    keys=[f's{i}' for i in range(6)]+[f't{i}' for i in range(8)]
    five=keys[:6]+['t0','t2','t4','t6','t7']
    four=keys[:6]+['t6','t0','t4','t2']
    a=select_noise(bank,keys,five);b=select_noise(bank,keys,four)
    assert a.dtype==b.dtype==torch.bfloat16
    assert torch.equal(a[:,:,:6],b[:,:,:6])
    for key in four[6:]:
        assert noise_digest(a[:,:,five.index(key)])==noise_digest(b[:,:,four.index(key)])
    assert torch.equal(select_noise(bank,keys,keys),bank)


@pytest.mark.parametrize('keys,requested', [(['a','a'],['a']),(['a','b'],['c']),(['a','b'],['a','a'])])
def test_invalid_or_ambiguous_noise_mapping_is_rejected(keys,requested):
    with pytest.raises(ValueError):select_noise(torch.zeros(1,2,2,2,2),keys,requested)
