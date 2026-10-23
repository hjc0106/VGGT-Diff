"""Read-only training audit; repair branch loss metadata from saved alpha arrays."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import vggtdiff_batch678_regularization_ablation as study
from analyze_batch678_reconstruction import outline_diagnostics


def main():
    study.base.OUT=study.BASELINE
    baseline_class=study.ORIGINAL_TRAINER()
    report={'branches':{},'diagnostic_fix':'Branch-aware outline normalization and weight; main area/PSNR metrics unchanged'}
    confs=[]
    for label,root in [('baseline',study.BASELINE),('outline',study.OUT/'outline'),('needle',study.OUT/'needle')]:
        study.base.OUT=root;study.base.validate()
        conf=OmegaConf.to_container(study.base.load_config(),resolve=True)
        for key in ['path','out_dir']:conf.pop(key)
        conf['initialization'].pop('fused_point_cloud_path');conf['export_ply'].pop('path');confs.append(conf)
        for name in ['inputs.json','sampling_plan.json','evaluation/cameras.json','shared/initial_points.ply']:
            assert study.base.chain.digest(root/name)==study.base.chain.digest(study.BASELINE/name)
        result=study.base.read(root/'training/result.json')
        assert result['step']==15000 and len(result['actual_sample_counts'])==24
        assert set(result['actual_sample_counts'].values())=={625}
        assert result['initial_state_sha256']==study.base.read(study.BASELINE/'training/initial_state.json')['sha256']
        assert result['ply_sha256']==study.base.chain.digest(root/'model.ply')
        if label!='baseline':
            assert (root/'implementation'/Path(study.__file__).name).read_bytes()==Path(study.__file__).read_bytes()
        rows=study.base.read(root/'inputs.json')['frames']
        settings=study.SETTINGS.get(label,dict(outline_foreground_normalized=False,outline_weight=.2))
        diag_path=root/'evaluation/analysis_v1/diagnostics.json';diag=study.base.read(diag_path)
        factors=[]
        for row,score in zip(rows,diag['per_frame']):
            assert row['angle']==score['angle']
            mask=np.asarray(Image.open(row['mask']))>127
            alpha=np.load(root/'evaluation/analysis_v1/training_alpha_overlays'/f"{score['index']:02d}_alpha.npz")['alpha']
            values=outline_diagnostics(alpha,mask,settings)
            # Verify CPU diagnostics against the actual differentiable training term.
            tmask=torch.from_numpy(mask.astype('float32'))[None,:,:,None]
            talpha=torch.from_numpy(alpha)[None,:,:,None]
            loss,_=study.extra_terms(talpha,tmask,torch.ones((1,3)),torch.ones(1),settings|{'needle_ratio':4.})
            assert math.isclose(float(loss),values['training_outside_opacity_loss'],rel_tol=1e-5,abs_tol=1e-7)
            assert math.isclose(float(alpha[~mask].sum()/mask.sum()),score['outside_opacity_over_mask_area'],rel_tol=1e-6)
            for threshold in [.05,.5]:
                outside=float(((alpha>threshold)&~mask).sum()/mask.sum())
                missing=float((mask&~(alpha>threshold)).sum()/mask.sum())
                assert outside==score[f'outside_area_over_mask_{threshold}']
                assert missing==score[f'missing_mask_fraction_{threshold}']
            old=values['training_outside_opacity_background_normalized_reference']
            factors.append(values['training_outside_opacity_contribution']/max(.2*old,1e-20))
            if label!='baseline':score.update(values)
        if label!='baseline':
            for batch,group in diag['by_batch'].items():
                sub=[r for r in diag['per_frame'] if r['batch']==int(batch)]
                for k in values:
                    if k!='training_outside_opacity_denominator':group[k]=float(np.mean([r[k] for r in sub]))
            study.base.save(diag_path,diag)
        report['branches'][label]=dict(frozen_hashes_passed=True,config_equal_except_paths=True,
            steps=result['step'],updates_per_image=625,initial_state=result['initial_state_sha256'],
            effective_outline_strength_vs_original_on_same_alpha=dict(min=min(factors),max=max(factors),endpoint=factors[-1]),
            endpoint_actual_outline_loss=values['training_outside_opacity_loss'],
            endpoint_actual_outline_contribution=values['training_outside_opacity_contribution'])
    assert confs[0]==confs[1]==confs[2]
    # Exercise the actual get_losses overrides, including gradients and coefficients.
    torch.manual_seed(123)
    # Production SSIM is a fused CUDA kernel.
    rgb=torch.rand(1,32,32,3,device='cuda',requires_grad=True)
    alpha=torch.rand(1,32,32,1,device='cuda',requires_grad=True)
    mask=torch.zeros_like(alpha);mask[:,8:24,8:24]=1
    scales=torch.tensor([[1.,1.,8.],[1.,8.,8.],[1.,1.,1.]],device='cuda',requires_grad=True)
    opacity=torch.tensor([.8,.5,.2],device='cuda',requires_grad=True)
    model=SimpleNamespace(get_scale=lambda:scales,get_density=lambda:opacity)
    dummy=SimpleNamespace(sample_counts=Counter(),model=model)
    batch=SimpleNamespace(image_name='00.png',rgb_gt=torch.rand_like(rgb),mask=mask)
    outputs=dict(pred_rgb=rgb,pred_opacity=alpha)
    original=baseline_class.get_losses(dummy,batch,outputs)
    for branch in ['outline','needle']:
        study.configure(branch);cls=study.base.trainer_class()
        losses=cls.get_losses(dummy,batch,outputs)
        outline,needle=study.extra_terms(alpha,mask,scales,opacity,study.SETTINGS[branch])
        expected=original['total_loss']-.2*original['opacity_loss']+study.SETTINGS[branch]['outline_weight']*outline+study.SETTINGS[branch]['needle_weight']*needle
        torch.testing.assert_close(losses['total_loss'],expected)
        actual_grads=torch.autograd.grad(losses['total_loss'],[rgb,alpha,scales,opacity],retain_graph=True)
        expected_grads=torch.autograd.grad(expected,[rgb,alpha,scales,opacity],retain_graph=True)
        for a,b in zip(actual_grads,expected_grads):torch.testing.assert_close(a,b)
    study.checks()
    report['actual_get_losses_and_gradients']='passed on controlled synthetic batch'
    report['saved_alpha_metrics_recomputed']='all 24 views x 3 models passed'
    study.base.save(study.OUT/'comparison/code_audit.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
