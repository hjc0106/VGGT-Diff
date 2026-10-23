"""Fixed-input outline/needle ablations; frozen baseline implementation reused."""
from pathlib import Path
import argparse
import math
import numpy as np
import vggtdiff_batch678_reconstruct as base

BASELINE=base.OUT
OUT=base.chain.BASE/'vggtdiff_batch678_regularization_ablation_v1'
SETTINGS={
    'outline':dict(outline_foreground_normalized=True,outline_weight=.05,needle_weight=0.,needle_ratio=4.),
    'needle':dict(outline_foreground_normalized=False,outline_weight=.2,needle_weight=.01,needle_ratio=4.),
    'combined':dict(outline_foreground_normalized=True,outline_weight=.05,needle_weight=.01,needle_ratio=4.)}
ORIGINAL_TRAINER=base.trainer_class


def extra_terms(alpha,mask,scales,opacity,settings):
    import torch
    import torch.nn.functional as F
    expanded=F.max_pool2d(mask.permute(0,3,1,2),3,1,1).permute(0,2,3,1)
    outside=1-expanded
    denominator=mask.sum() if settings['outline_foreground_normalized'] else outside.sum()
    outline=(alpha*outside).sum()/denominator.clamp_min(1)
    sorted_scales=scales.sort(dim=-1).values
    ratio=sorted_scales[:,-1]/sorted_scales[:,-2].clamp_min(1e-8)
    needle=(torch.relu(torch.log(ratio)-math.log(settings['needle_ratio']))*opacity).sum()/opacity.sum().clamp_min(1e-6)
    return outline,needle


def prepare(branch):
    from omegaconf import OmegaConf
    base.OUT=BASELINE;base.validate()
    root=OUT/branch
    if (root/'experiment.json').exists():
        base.OUT=root;base.validate();return
    root.mkdir(parents=True,exist_ok=True)
    (root/'dataset').symlink_to(BASELINE/'dataset',target_is_directory=True)
    for name in ['inputs.json','sampling_plan.json','evaluation/cameras.json','mask_review.json','shared/initial_points.ply','shared/initialization.json']:
        base.chain.frozen_copy(BASELINE/name,root/name)
    conf=base.load_config()
    before=OmegaConf.to_container(conf,resolve=True)
    conf.path=str(root/'dataset');conf.out_dir=str(root/'training')
    conf.initialization.fused_point_cloud_path=str(root/'shared/initial_points.ply');conf.export_ply.path=str(root/'model.ply')
    after=OmegaConf.to_container(conf,resolve=True)
    for obj in [before,after]:
        for key in ['path','out_dir']:obj.pop(key)
        obj['initialization'].pop('fused_point_cloud_path');obj['export_ply'].pop('path')
    assert before==after,'Unexpected non-output configuration change'
    OmegaConf.save(conf,root/'training_config.yaml')
    base.save(root/'loss_settings.json',SETTINGS[branch])
    experiment=base.read(BASELINE/'experiment.json')
    files=[Path(r['path']) for r in experiment['frozen_files']]
    files += [Path(__file__).resolve(),BASELINE/'experiment.json',BASELINE/'training/result.json',
        OUT/'protocol.json',root/'loss_settings.json',root/'training_config.yaml']
    files += [root/n for n in ['inputs.json','sampling_plan.json','evaluation/cameras.json','shared/initial_points.ply','shared/initialization.json']]
    experiment.update(branch=branch,baseline=str(BASELINE),loss_settings=SETTINGS[branch],
        loss='Same baseline photometric/anisotropy losses; only declared outline/needle terms change',
        frozen_files=[dict(path=str(p),sha256=base.chain.digest(p)) for p in dict.fromkeys(files)])
    base.save(root/'experiment.json',experiment)
    base.chain.frozen_copy(Path(__file__).resolve(),root/'implementation'/Path(__file__).name)
    base.OUT=root;base.validate();base.save(root/'status.json',dict(state='prepared',steps=15000))
    print('PREPARED',branch,root,flush=True)


def configure(branch):
    base.OUT=OUT/branch
    settings=SETTINGS[branch]
    def build():
        import torch
        parent=ORIGINAL_TRAINER()
        from threedgrut.model.losses import ssim
        from car15_object_loss import object_losses
        class AblationTrainer(parent):
            def setup_training(self,conf,model,dataset):
                super().setup_training(conf,model,dataset)
                expected=base.read(BASELINE/'training/initial_state.json')['sha256']
                assert base.prior.model_hash(model)==expected,'Initial model differs from baseline'
            def get_losses(self,batch,outputs):
                self.sample_counts[Path(batch.image_name).name]+=1
                l1,ss,_=object_losses(outputs['pred_rgb'],batch.rgb_gt,outputs['pred_opacity'],batch.mask,ssim_fn=ssim,dilate=3)
                scales,opacity=self.model.get_scale(),self.model.get_density().reshape(-1)
                ratio=scales.amax(-1)/scales.amin(-1).clamp_min(1e-8)
                aniso=(torch.relu(torch.log(ratio)-math.log(16))*opacity).sum()/opacity.sum().clamp_min(1e-6)
                outline,needle=extra_terms(outputs['pred_opacity'],batch.mask,scales,opacity,settings)
                total=.8*l1+.2*ss+settings['outline_weight']*outline+.03*aniso+settings['needle_weight']*needle
                zero=total.new_zeros(())
                return dict(total_loss=total,l1_loss=l1,l2_loss=zero,ssim_loss=ss,opacity_loss=outline,
                    scale_loss=aniso,needle_loss=needle,lpips_loss=zero)
        return AblationTrainer
    base.trainer_class=build


def checks():
    import torch
    mask=torch.zeros((1,9,9,1));mask[:,3:6,3:6]=1
    alpha=torch.full_like(mask,.3,requires_grad=True)
    scales=torch.tensor([[1.,1.,8.],[1.,8.,8.],[1.,1.,1.]],requires_grad=True)
    opacity=torch.ones(3)
    out_b,needle=extra_terms(alpha,mask,scales,opacity,SETTINGS['outline'])
    out_c,_=extra_terms(alpha,mask,scales,opacity,SETTINGS['needle'])
    torch.testing.assert_close(out_b,torch.tensor(.3*56/9))
    torch.testing.assert_close(out_c,torch.tensor(.3))
    torch.testing.assert_close(needle,torch.tensor(math.log(2)/3))
    (out_b+needle).backward();assert torch.isfinite(alpha.grad).all() and torch.isfinite(scales.grad).all()
    # Disabled-needle baseline must retain the original outline term exactly.
    from car15_object_loss import object_losses
    rgb=torch.zeros((1,9,9,3))
    _,_,old=object_losses(rgb,rgb,alpha,mask,dilate=3)
    torch.testing.assert_close(out_c,old)
    print('LOSS_CHECKS_PASSED',flush=True)


def init_protocol():
    OUT.mkdir(exist_ok=True)
    p=OUT/'protocol.json'
    protocol=dict(baseline=str(BASELINE),settings=SETTINGS,seed=20261021,steps=15000,updates_per_image=625,
        fixed='Images,masks,W2C,K,initial state,sampling order,optimizer,LR,SH,densify/prune,render cameras',
        primary='Endpoint354.375: outside area alpha>.5 target about8%; all24 PSNR drop<=1dB, missing-mask increase<=.005',
        secondary='Outside alpha>.05, integrated opacity, IoU, shape distributions, all5 video paths',
        targets_are='Exploratory engineering screening criteria, not general geometry acceptance lines',
        phase2='Run combined only if both single interventions have useful gains; no hyperparameter retuning in phase1',
        scope='Generated-target fitting and visual stability; not independent realGT')
    if p.exists():assert base.read(p)==protocol
    else:base.save(p,protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','checks','preflight','train','render','run']);p.add_argument('--branch',choices=list(SETTINGS));a=p.parse_args()
    if a.action=='checks':checks()
    else:
        init_protocol()
        if a.action=='prepare':
            for b in ([a.branch] if a.branch else ['outline','needle']):prepare(b)
        else:
            if not a.branch:p.error('--branch required')
            configure(a.branch)
            if a.action=='preflight':base.preflight()
            elif a.action=='train':base.train()
            elif a.action=='render':base.render_results()
            else:
                base.validate()
                if not (base.OUT/'training/result.json').exists():base.train()
                base.render_results()
