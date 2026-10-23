"""Instrument one denoising step; never overwrite the completed 50-step pilot."""
from experiment_paths import checkpoint_path, omega_path, base_model_path
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vggtdiff_white_car import OUT, REFS, PILOT, digest
import vggtdiff.omega as omega_module
from vggtdiff import VGGTDiff


def stats(tensor):
    t=tensor.detach().float()
    return dict(shape=list(t.shape),finite=bool(torch.isfinite(t).all()),
                mean=float(t.mean()),std=float(t.std()),minimum=float(t.min()),maximum=float(t.max()))


def main():
    out=OUT/'audit';out.mkdir(exist_ok=True)
    report={'purpose':'one-step forward diagnostic, not a replacement 50-step repair run',
            'source_indices':REFS,'targets':PILOT,'full_pilot_unchanged':True}
    images=[Image.open(OUT/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
    trajectory=json.loads((OUT/'trajectory.json').read_text())
    w2c=np.asarray(trajectory['source_w2c']+[trajectory['target_w2c'][i] for i in PILOT],dtype=np.float32)
    k=np.asarray(trajectory['source_intrinsics']+[trajectory['target_intrinsics'][i] for i in PILOT],dtype=np.float32)
    report['source_file_mapping']=[{'source_slot':j,'trajectory_frame':i,
        'matches_padded_baseline':digest(OUT/'source_views'/f'{j:02d}.png')==digest(OUT/'baseline/letterboxed'/f'{i:05d}.png')} for j,i in enumerate(REFS)]
    original_similarity=omega_module._similarity_transform
    def similarity(source,target):
        result=original_similarity(source,target)
        scale,rotation,translation=result
        fitted=scale*(np.asarray(source) @ rotation.T)+translation
        rms=float(np.sqrt(np.square(fitted-target).sum(axis=-1).mean()))
        baseline=float(np.median(np.linalg.norm(np.asarray(target)[:,None]-np.asarray(target)[None],axis=-1)[np.triu_indices(6,1)]))
        report['omega_camera_center_alignment']={'rms':rms,'median_source_baseline':baseline,'relative_rms':rms/baseline,'scale':float(scale)}
        return result
    omega_module._similarity_transform=similarity
    model=VGGTDiff(str(checkpoint_path()),
        str(omega_path()),
        base_model_dir=str(base_model_path()),
        device='cuda',vram_limit_gib=20,training_mode='nvs',load_full_checkpoint_directly=True)
    report['constructor_training_flag']=model.pipe.dit.training
    # Reproduce the historical pilot flag, then compare against explicit eval.
    model.pipe.dit.train()
    report['dit_training_flag_before']=model.pipe.dit.training
    report['dropout_or_batchnorm_modules']=[name for name,m in model.pipe.dit.named_modules()
        if isinstance(m,(torch.nn.modules.dropout._DropoutNd,torch.nn.modules.batchnorm._BatchNorm))]
    original_omega=model.omega.forward
    def geometry(*args,**kwargs):
        result=original_omega(*args,**kwargs)
        report['geometry']={name:stats(value) for name,value in result.items()}
        return result
    model.omega.forward=geometry
    original_fn=model.pipe.model_fn
    def instrumented(**kwargs):
        report['model_inputs']={name:stats(kwargs[name]) for name in ['latents','y','raymap','context','omega_tokens','omega_xyz','omega_confidence']}
        assert tuple(kwargs['latents'].shape)==(1,16,14,60,104)
        assert tuple(kwargs['y'].shape)==(1,20,14,60,104)
        report['num_output_frames']=kwargs['num_output_frames']
        first=original_fn(**kwargs)
        report['velocity']=stats(first[:,:,-8:])
        model.pipe.dit.eval()
        second=original_fn(**kwargs)
        difference=(first.float()-second.float()).abs()
        report['train_vs_eval_difference']={'max_abs':float(difference.max()),'mean_abs':float(difference.mean()),'exact_equal':bool(torch.equal(first,second))}
        repeated=original_fn(**kwargs)
        difference=(repeated.float()-second.float()).abs()
        report['eval_vs_eval_repeat_difference']={'max_abs':float(difference.max()),'mean_abs':float(difference.mean()),'exact_equal':bool(torch.equal(repeated,second))}
        no_geometry=original_fn(**dict(kwargs,omega_condition_scale=0.0))
        diff=(second[:,:,-8:].float()-no_geometry[:,:,-8:].float()).abs()
        report['geometry_condition_ablation']={'mean_abs_velocity_change':float(diff.mean()),'max_abs_velocity_change':float(diff.max())}
        (out/'forward_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print('FORWARD_AUDIT',json.dumps(report),flush=True)
        return second
    model.pipe.model_fn=instrumented
    original_step=model.pipe.scheduler.step
    def scheduler_step(velocity,timestep,latents,*args,**kwargs):
        result=original_step(velocity,timestep,latents,*args,**kwargs)
        d=(result[:,:,-8:].float()-latents[:,:,-8:].float()).abs()
        report['target_latent_update']={'mean_abs_change':float(d.mean()),'max_abs_change':float(d.max()),'finite':bool(torch.isfinite(result).all())}
        return result
    model.pipe.scheduler.step=scheduler_step
    result=model.generate(images,w2c,k,height=480,width=832,steps=1,seed=20260823)
    report['diagnostic_output_count']=len(result)
    report['diagnostic_output_sizes']=[list(im.size) for im in result]
    (out/'forward_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
