"""Prepare/train/render the24 generated views from batches6,new7,real6-conditioned8.

prepare and preflight never run optimizer steps. run trains then renders videos.
"""
from pathlib import Path
from collections import Counter
import argparse
import math
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp
import evaluate_batch07_two_sided_bridge as bridge
import vggtdiff_real8_generated8_reconstruct as prior

chain=bridge.job.chain
OUT=chain.BASE/'vggtdiff_batch06_07bridge_08_reconstruction_v1'
SEED=20261021
EXPOSURES=625
STEPS=24*EXPOSURES


def read(p):return chain.read(p)
def save(p,v):return chain.save(p,v)


def config(output_dir=None):
    import vggtdiff_car15_train
    from data_processing.threedgrut_training import compose_3dgrut_config,DEFAULT_THREEDGRUT_CONFIG_DIR
    root=output_dir or OUT/'training'
    overrides=[f'path={OUT/"dataset"}',f'out_dir={root}','experiment_name=batch678',
        'initialization=fused_point_cloud',f'initialization.fused_point_cloud_path={OUT/"shared/initial_points.ply"}',
        f'n_iterations={STEPS}','num_workers=0','enable_writer=false','use_wandb=false','with_gui=false',
        'test_last=false','validate_first=false','compute_extra_metrics=false','export_ingp.enabled=false',
        'export_ply.enabled=true',f'export_ply.path={OUT/"model.ply"}',
        'dataset.test_split_interval=0','val_frequency=999999','checkpoint.iterations=[5000,10000,15000]',
        'model.progressive_training.max_n_features=1','model.progressive_training.increase_frequency=3000',
        'scheduler.positions.max_steps=15000','strategy.densify.frequency=900',
        'strategy.densify.start_iteration=1500','strategy.densify.end_iteration=9000',
        'strategy.prune.frequency=300','strategy.prune.start_iteration=1500','strategy.prune.end_iteration=13500',
        'strategy.reset_density.end_iteration=-1','+car15_stage=batch678']
    return compose_3dgrut_config('apps/colmap_3dgut',overrides,DEFAULT_THREEDGRUT_CONFIG_DIR)


def camera_sets(rows):
    angles=np.array([r['angle'] for r in rows]);cams=np.linalg.inv(np.array([r['w2c'] for r in rows]));ks=np.array([r['k'] for r in rows])
    slerp=Slerp(angles,Rotation.from_matrix(cams[:,:3,:3]))
    def interpolate(values):
        result=[]
        for a in values:
            c=np.eye(4);c[:3,:3]=slerp(a).as_matrix();c[:3,3]=[np.interp(a,angles,cams[:,j,3]) for j in range(3)]
            k=np.array([[np.interp(a,angles,ks[:,i,j]) for j in range(3)] for i in range(3)])
            result.append(dict(angle=float(a),w2c=np.linalg.inv(c).tolist(),k=k.tolist()))
        return result
    local=interpolate(np.linspace(angles[0],angles[-1],128))
    left=interpolate(np.linspace(angles[6],angles[9],64))
    right=interpolate(np.linspace(angles[14],angles[17],64))
    g=read(bridge.job.previous.GEOMETRY);up=np.array(g['up']);up/=np.linalg.norm(up)
    elevated=[]
    for r in local:
        c=np.linalg.inv(r['w2c']);c[:3,3]+=.025*up;elevated.append(dict(r,w2c=np.linalg.inv(c).tolist()))
    orbit=bridge.job.previous.continuous_orbit(rows[7]['w2c'],rows[7]['k'],rows[7]['angle'],g['center'],g['up'],np.arange(128)*360/128)
    whole=[dict(angle=a,w2c=w,k=k) for a,w,k in zip(orbit['target_angles_deg'],orbit['target_w2c'],orbit['target_intrinsics'])]
    # Interpolation must reproduce every supplied camera, including both seams.
    for a,b in zip(rows,interpolate(angles)):
        np.testing.assert_allclose(a['w2c'],b['w2c'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(a['k'],b['k'],atol=1e-12,rtol=0)
    return dict(supervised_arc=local,boundary67=left,boundary78=right,elevated_arc=elevated,orbit360=whole)


def prepare():
    from omegaconf import OmegaConf
    bridge.validate()
    if (OUT/'experiment.json').exists():
        validate();print('ALREADY_PREPARED',OUT,flush=True);return
    OUT.mkdir(exist_ok=True)
    source=[r for r in read(bridge.OUT/'frames.json')['frames'] if r['variant']=='new7']
    assert len(source)==24 and [r['index'] for r in source]==list(range(40,64))
    rows=[]
    for i,r in enumerate(source):
        image=Path(r['image']);mask=Path(r['mask'])
        assert chain.digest(image)==r['sha256'] and chain.digest(mask)==r['mask_sha256']
        assert Image.open(image).size==Image.open(mask).size==(832,480)
        rows.append(dict(name=f'{i:02d}.png',batch=r['batch'],source_sequence_index=r['index'],
            angle=r['azimuth_deg'],image=str(image),mask=str(mask),sha256=r['sha256'],mask_sha256=r['mask_sha256'],
            w2c=r['T'],k=r['K'],role='generated'))
    assert [r['batch'] for r in rows]==[6]*8+[7]*8+[8]*8
    assert all(rows[i]['angle']<rows[i+1]['angle'] for i in range(23))
    for name in ['initial_points.ply','initialization.json']:
        chain.frozen_copy(prior.OUT/'shared'/name,OUT/'shared'/name)
    previous_out=prior.ab.OUT
    try:
        prior.ab.OUT=OUT
        prior.ab.write_colmap(OUT/'dataset',rows)
    finally:prior.ab.OUT=previous_out
    save(OUT/'inputs.json',dict(frames=rows,width=832,height=480,coordinates='Original Omega world, OpenCV W2C',direct_real_training_images=0))
    save(OUT/'evaluation/cameras.json',camera_sets(rows))
    rng=np.random.default_rng(SEED);stems=np.concatenate([rng.permutation(24) for _ in range(EXPOSURES)])
    save(OUT/'sampling_plan.json',dict(seed=SEED,stems=stems.tolist(),counts=dict(Counter(map(str,stems))),steps=STEPS,updates_per_image=EXPOSURES))
    conf=config();OmegaConf.save(conf,OUT/'training_config.yaml')
    init=read(OUT/'shared/initialization.json');old=read(prior.OUT/'experiment.json')
    files=[Path(__file__).resolve(),Path(bridge.__file__).resolve(),Path(prior.__file__).resolve(),
        Path(prior.ab.__file__).resolve(),bridge.OUT/'manifest.json',bridge.OUT/'summary.json',
        prior.OUT/'experiment.json',OUT/'inputs.json',OUT/'evaluation/cameras.json',OUT/'sampling_plan.json',OUT/'training_config.yaml',
        OUT/'shared/initial_points.ply',OUT/'shared/initialization.json']
    files += list((OUT/'dataset/sparse/0').glob('*'))+[OUT/'dataset/point_cloud.ply']
    files += [Path(r[k]) for r in rows for k in ['image','mask']]
    for filename in ['vggtdiff_car15_train.py','car15_gsplat_backend.py','car15_object_loss.py','vggtdiff_selfconsistent_silhouette.py']:
        files.append(chain.REPO/'scripts'/filename)
    experiment=dict(images=24,batches=[6,7,8],training_seed=SEED,steps=STEPS,updates_per_image=EXPOSURES,
        coverage_deg=[rows[0]['angle'],rows[-1]['angle']],scene_extent=old['scene_extent'],
        cameras='Frozen per-image W2C/K, no camera optimization; batch6 actual matched-camera angles preserved',
        initialization='Same untrained real-derived Omega geometry prior as prior8 reconstruction:17112 surface+4000 low-opacity volume seeds. No trained checkpoint warmup.',
        loss='0.8 foregroundL1 +0.2 foreground-cropSSIM +0.2 outside opacity(dilate3) +0.03 opacity-weighted anisotropy above ratio16; all24 image weights1',
        schedule='625 shuffled24-view epochs; total15000, exactly625 updates/image. PositionLR, SH increase and densify/prune event schedule scaled3x from prior5000-step8-view run.',
        backend='3DGRUT Gaussian model/optimizer/strategy, gsplat3DGS rasterizer; 100k Gaussian cap',
        videos='supervised_arc128, boundary67/78 each64, elevated_arc128, orbit360128;832x480,12fps',
        limitations='All24 training images generated; initial geometry real-derived. Single training seed; training-fit scores are not independentGT. Supervision covers about130deg, so360 render contains unsupported extrapolation.',
        frozen_files=[dict(path=str(p),sha256=chain.digest(p)) for p in dict.fromkeys(files)])
    save(OUT/'experiment.json',experiment)
    for p in [Path(__file__).resolve(),Path(prior.__file__).resolve()]:chain.frozen_copy(p,OUT/'implementation'/p.name)
    chain.montage([Path(r['image']) for r in rows],[f"{i:02d} | B{r['batch']} | {r['angle']:.3f} deg" for i,r in enumerate(rows)],OUT/'inputs24.jpg',4,(416,240))
    overlays=[]
    for i,r in enumerate(rows):
        rgb=np.asarray(Image.open(r['image']).convert('RGB')).copy();mask=np.asarray(Image.open(r['mask']))>127
        rgb[mask]=(.55*rgb[mask]+.45*np.array([0,255,0])).astype('uint8');p=OUT/'mask_overlays'/f'{i:02d}.jpg';p.parent.mkdir(exist_ok=True);Image.fromarray(rgb).save(p);overlays.append(p)
    chain.montage(overlays,[f"{i:02d} | B{r['batch']} | {r['angle']:.3f} deg" for i,r in enumerate(rows)],OUT/'masks24.jpg',4,(416,240))
    validate();save(OUT/'status.json',dict(state='prepared',training_started=False,images=24,steps=STEPS))
    print('PREPARED',OUT,flush=True)


def validate():
    for r in read(OUT/'experiment.json')['frozen_files']:assert chain.digest(Path(r['path']))==r['sha256'],r['path']
    for r in read(OUT/'inputs.json')['frames']:
        assert chain.digest(OUT/'dataset/images'/r['name'])==r['sha256']
        assert chain.digest(OUT/'dataset/images'/f"{Path(r['name']).stem}_mask.png")==r['mask_sha256']


def load_config(output_dir=None):
    import vggtdiff_car15_train
    from omegaconf import OmegaConf
    from data_processing.threedgrut_training import register_3dgrut_resolvers
    register_3dgrut_resolvers()
    conf=OmegaConf.load(OUT/'training_config.yaml')
    if output_dir is not None:conf.out_dir=str(output_dir)
    return conf


def trainer_class():
    import torch
    import vggtdiff_car15_train as old
    from threedgrut.trainer import Trainer3DGRUT
    from threedgrut.datasets.utils import MultiEpochsDataLoader
    from threedgrut.model.losses import ssim
    from car15_object_loss import object_losses
    class BridgeTrainer(old.ObjectTrainer):
        def init_dataloaders(self,conf):
            Trainer3DGRUT.init_dataloaders(self,conf)
            names=[Path(p).name for p in self.train_dataset.image_paths];assert sorted(names)==[f'{i:02d}.png' for i in range(24)]
            lookup={int(Path(n).stem):i for i,n in enumerate(names)};stems=read(OUT/'sampling_plan.json')['stems'];schedule=[lookup[i] for i in stems]
            class FixedSampler(torch.utils.data.Sampler):
                def __iter__(self):return iter(schedule)
                def __len__(self):return len(schedule)
            self.train_dataloader=MultiEpochsDataLoader(self.train_dataset,batch_size=1,num_workers=0,sampler=FixedSampler(),pin_memory=False)
            self.sample_counts=Counter()
        def init_scene_extents(self,dataset):
            self.scene_extent=read(OUT/'experiment.json')['scene_extent']
            self.scene_bbox=tuple(torch.tensor(v,device=self.device,dtype=torch.float32) for v in read(OUT/'shared/initialization.json')['bounds_world'])
        def setup_training(self,conf,model,train_dataset):
            Trainer3DGRUT.setup_training(self,conf,model,train_dataset)
            n=read(OUT/'shared/initialization.json')['real_surface_points']
            with torch.no_grad():model.density[n:]=float(np.log(.01/.99))
        def get_losses(self,batch,outputs):
            self.sample_counts[Path(batch.image_name).name]+=1
            l1,ss,alpha=object_losses(outputs['pred_rgb'],batch.rgb_gt,outputs['pred_opacity'],batch.mask,ssim_fn=ssim,dilate=3)
            scales,opacity=self.model.get_scale(),self.model.get_density().reshape(-1)
            ratio=scales.amax(-1)/scales.amin(-1).clamp_min(1e-8)
            aniso=(torch.relu(torch.log(ratio)-math.log(16))*opacity).sum()/opacity.sum().clamp_min(1e-6)
            total=.8*l1+.2*ss+.2*alpha+.03*aniso;zero=total.new_zeros(())
            return dict(total_loss=total,l1_loss=l1,l2_loss=zero,ssim_loss=ss,opacity_loss=alpha,scale_loss=aniso,lpips_loss=zero)
    return BridgeTrainer


def preflight():
    import torch
    validate();torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    trainer=trainer_class()(load_config(OUT/'preflight'))
    assert trainer.global_step==0 and trainer.model.num_gaussians==21112
    initial_hash=prior.model_hash(trainer.model);checks=[]
    for index in [0,7,8,15,16,23]:
        from vggtdiff_selfconsistent_silhouette import render
        rgb,alpha,depth=render(trainer.model,read(OUT/'inputs.json')['frames'][index])
        assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all()
        assert float(alpha.max())>0
        checks.append(dict(index=index,alpha_max=float(alpha.max()),shape=list(rgb.shape)))
    raw_batch=next(iter(trainer.train_dataloader))
    gpu_batch=trainer.train_dataset.get_gpu_batch_with_intrinsics(raw_batch)
    outputs=trainer.model(gpu_batch,train=True)
    losses=trainer.get_losses(gpu_batch,outputs)
    assert all(bool(torch.isfinite(v)) for v in losses.values())
    assert losses['total_loss'].requires_grad
    assert trainer.global_step==0 and prior.model_hash(trainer.model)==initial_hash
    validate()
    save(OUT/'preflight.json',dict(state='passed',global_step=trainer.global_step,optimizer_steps=0,
        images=len(trainer.train_dataset),initial_gaussians=trainer.model.num_gaussians,initial_state_sha256=initial_hash,
        prior8_initial_hash=read(prior.OUT/'generated8/training/initial_state.json')['sha256'],render_checks=checks,
        training_forward_loss={k:float(v.detach()) for k,v in losses.items()},training_config_loaded_in_fresh_process=True))
    print('PREFLIGHT_PASSED',OUT,flush=True)


def train():
    import torch
    from omegaconf import OmegaConf
    validate()
    if (OUT/'training/result.json').exists():raise FileExistsError('Training already complete; use render')
    if (OUT/'training/initial_state.json').exists():raise FileExistsError('Partial training exists; preserve it and use a separate experiment for a restart')
    assert read(OUT/'mask_review.json')['state']=='accepted'
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    conf=load_config();start=time.time();trainer=trainer_class()(conf);trainer.n_epochs=1
    assert trainer.global_step==0
    initial_hash=prior.model_hash(trainer.model)
    save(OUT/'training/initial_state.json',dict(sha256=initial_hash,gaussians=trainer.model.num_gaussians))
    save(OUT/'status.json',dict(state='training',steps=STEPS));trainer.run_training();trainer.save_checkpoint()
    assert trainer.global_step==STEPS and len(trainer.sample_counts)==24 and set(trainer.sample_counts.values())=={EXPOSURES}
    checkpoint=Path(trainer.tracking.output_dir)/f'ours_{STEPS}'/f'ckpt_{STEPS}.pt'
    assert checkpoint.exists() and (OUT/'model.ply').exists();validate()
    save(OUT/'training/result.json',dict(step=STEPS,checkpoint=str(checkpoint),elapsed_seconds=time.time()-start,
        gaussians=trainer.model.num_gaussians,initial_state_sha256=initial_hash,ply_sha256=chain.digest(OUT/'model.ply'),actual_sample_counts=dict(trainer.sample_counts)))
    save(OUT/'status.json',dict(state='trained_pending_render',steps=STEPS));print('TRAINED',checkpoint,flush=True)


def render_results():
    import torch
    from omegaconf import OmegaConf
    from plyfile import PlyData
    import vggtdiff_car15_train
    from threedgrut.model.model import MixtureOfGaussians
    from vggtdiff_selfconsistent_silhouette import render
    from vggtdiff_car15_evaluate import metrics
    validate();torch.set_num_threads(4)
    result=read(OUT/'training/result.json');assert chain.digest(OUT/'model.ply')==result['ply_sha256']
    model=MixtureOfGaussians(load_config())
    model.init_from_checkpoint(torch.load(result['checkpoint'],map_location='cuda',weights_only=False),setup_optimizer=False);model.eval()
    report=[]
    for i,r in enumerate(read(OUT/'inputs.json')['frames']):
        rgb,alpha,depth=render(model,r);assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all()
        gt=np.asarray(Image.open(r['image']).convert('RGB'),float)/255;mask=np.asarray(Image.open(r['mask']))>127
        row=dict(index=i,batch=r['batch'],angle=r['angle'],**metrics(rgb,alpha,gt,mask),outside_opacity_over_gt_area=float(alpha[~mask].sum()/mask.sum()))
        report.append(row);canvas=Image.new('RGB',(1664,512),(22,22,22));draw=ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(np.uint8(np.clip(gt*mask[:,:,None],0,1)*255)),(0,32));canvas.paste(Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)),(832,32))
        draw.text((8,10),f"Generated supervision | B{r['batch']} | {r['angle']:.3f}deg",fill='white');draw.text((840,10),'3DGS reconstruction | training camera',fill='white')
        p=OUT/'evaluation/training_comparison'/f'{i:03d}.png';p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
    sets=read(OUT/'evaluation/cameras.json')
    for name,rows in sets.items():
        for i,r in enumerate(rows):
            rgb,alpha,depth=render(model,r);assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all()
            prior.ab.write_rgb(OUT/'evaluation'/name/f'{i:03d}.png',rgb)
            if name!='orbit360':prior.ab.write_rgb(OUT/'evaluation'/(name+'_alpha')/f'{i:03d}.png',np.repeat(alpha[:,:,None],3,axis=2))
        make_video(OUT/'evaluation'/name,OUT/'evaluation'/f'{name}.mp4')
        prior.ab.montage([OUT/'evaluation'/name/f'{i:03d}.png' for i in np.linspace(0,len(rows)-1,8).astype(int)],
            [f"{name} | {rows[i]['angle']:.2f}deg" for i in np.linspace(0,len(rows)-1,8).astype(int)],OUT/'evaluation'/f'{name}_contact.jpg',4,(416,240))
        print('RENDERED',name,len(rows),flush=True)
    make_video(OUT/'evaluation/training_comparison',OUT/'evaluation/training_comparison.mp4')
    ply=PlyData.read(OUT/'model.ply')['vertex'].data;assert all(np.isfinite(ply[k]).all() for k in ply.dtype.names)
    scale=np.exp(np.column_stack([ply[f'scale_{i}'] for i in range(3)]).astype(float));ratio=scale.max(1)/np.maximum(scale.min(1),1e-12);opacity=1/(1+np.exp(-ply['opacity'].astype(float)))
    exported=MixtureOfGaussians(load_config());exported.init_from_ply(str(OUT/'model.ply'),init_model=False);exported.eval();errors=[]
    for i in [0,32,64,96]:
        a,_,_=render(model,sets['orbit360'][i]);b,_,_=render(exported,sets['orbit360'][i]);errors.append(float(np.abs(a-b).mean()))
    assert max(errors)<1e-6
    save(OUT/'evaluation/summary.json',dict(training_fit_to_generated_inputs=report,
        model=dict(gaussians=len(ply),anisotropy_quantiles=np.quantile(ratio,[.5,.9,.99]).tolist(),opacity_fraction_ratio_above16=float(opacity[ratio>16].sum()/opacity.sum())),
        caution='Training fit is not independentGT; supervision covers about130deg, orbit360 includes unsupported views.'))
    save(OUT/'evaluation/audit.json',dict(state='passed',finite_ply_and_renders=True,ply_roundtrip_rgb_mae=errors,
        every_image_updates=EXPOSURES,total_updates=STEPS,quality_acceptance_threshold=None))
    validate();save(OUT/'status.json',dict(state='completed_pending_visual_review',steps=STEPS,evaluation=str(OUT/'evaluation')))
    print('RESULTS_READY',OUT/'evaluation',flush=True)


def make_video(directory,destination):
    subprocess.run(['ffmpeg','-y','-framerate','12','-i',str(directory/'%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(destination)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','check','preflight','train','render','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':validate();print('FROZEN_INPUTS_OK',OUT)
    elif a.action=='preflight':preflight()
    elif a.action=='train':train()
    elif a.action=='render':render_results()
    else:
        validate()
        if not (OUT/'training/result.json').exists():train()
        render_results()
