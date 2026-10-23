import numpy as np
import pytest
import torch
from PIL import Image

import vggtdiff.model as runtime


@pytest.mark.parametrize('mode,expected', [('nvs',14),('trajectory',None)])
def test_inference_forwards_correct_latent_count(monkeypatch, mode, expected):
    model=runtime.VGGTDiff.__new__(runtime.VGGTDiff)
    model.device=torch.device('cpu')
    model.dtype=torch.float32
    model.training_mode=mode
    monkeypatch.setattr(runtime,'plucker_rays',lambda *a,**kw:torch.zeros(14,384,2,2))
    model.omega=lambda **kw:{k:torch.zeros(1) for k in [
        'omega_tokens','omega_xyz','omega_confidence','omega_target_w2c','omega_target_intrinsics']}
    captured={}
    def pipeline(**kw):
        captured.update(kw)
        return [Image.new('RGB',(16,16)) for _ in range(14)]
    model.pipe=pipeline
    images=[Image.new('RGB',(16,16)) for _ in range(6)]
    output=model.generate(images,np.repeat(np.eye(4)[None],14,axis=0),
                          np.repeat(np.eye(3)[None],14,axis=0),height=16,width=16)
    assert captured['num_latent_frames']==expected
    assert captured['num_output_frames']==8
    assert len(output)==8
