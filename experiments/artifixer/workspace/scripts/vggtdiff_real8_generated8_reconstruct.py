"""Pure real8 versus generated8: frozen cameras, shared initialization, equal exposure.

Run prepare, train real8, train generated8, evaluate, compare in that order.
No diffusion reference images or photo-trained warmup enter training.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import vggtdiff_latest_coverage_ab as ab

OUT = ab.BASE/'vggtdiff_real8_vs_generated8_reconstruction_v1'
MATCH = ab.BASE/'crossview_geometry_matched_gt_v1'
REAL = ab.BASE/'crossview_geometry_dense_real8_v1'
SEED = 20261021
STEPS = 5000
SELECTED = 20261028


def prepare():
    if (OUT/'experiment.json').exists():
        raise FileExistsError('Experiment is already frozen')
    frames = ab.read(REAL/'frames.json')['frames']
    selected = MATCH/f'seed_{SELECTED}'
    results = ab.read(selected/'result.json')['frames']
    assert len(frames) == len(results) == 8
    for name in ['initial_points.ply', 'initialization.json']:
        ab.frozen_copy(ab.OUT/'shared'/name, OUT/'shared'/name)
    frozen = [OUT/'shared/initial_points.ply', OUT/'shared/initialization.json',
              REAL/'frames.json', MATCH/'protocol.json', MATCH/'trajectory.json', selected/'result.json']
    for branch in ['real8', 'generated8']:
        rows = []
        for i, (gt, gen) in enumerate(zip(frames, results)):
            assert gt['original_frame'] == gen['original_frame']
            image = Path(gt['image']) if branch == 'real8' else selected/'raw'/f'{i:02d}.png'
            mask = Path(gt['mask']) if branch == 'real8' else selected/'masks'/f'{i:02d}.png'
            assert ab.digest(image) == (gt['sha256'] if branch == 'real8' else gen['sha256'])
            assert Image.open(image).size == Image.open(mask).size == (832, 480)
            rows.append(dict(name=f'{i:02d}.png', original_frame=gt['original_frame'], angle=gt['azimuth_deg'],
                image=str(image), mask=str(mask), sha256=ab.digest(image), mask_sha256=ab.digest(mask),
                w2c=gt['T'], k=gt['K']))
            frozen += [image, mask]
        # The writer uses OUT only to find the shared, untrained point cloud.
        previous = ab.OUT
        try:
            ab.OUT = OUT
            ab.write_colmap(OUT/branch/'dataset', rows)
        finally:
            ab.OUT = previous
        ab.save(OUT/branch/'inputs.json', dict(frames=rows))
        frozen.append(OUT/branch/'inputs.json')
    a = ab.read(OUT/'real8/inputs.json')['frames']
    b = ab.read(OUT/'generated8/inputs.json')['frames']
    assert all(x['w2c'] == y['w2c'] and x['k'] == y['k'] for x,y in zip(a,b))
    ab.save(OUT/'experiment.json', dict(selected_seed=SELECTED,
        selection_rule='Lowest median epipolar error on the 14 pre-frozen <=15 degree pairs: seed20261028, 0.416205 px. Not best on every three-view metric.',
        training_seed=SEED, steps=STEPS, scene_extent=ab.read(ab.OUT/'experiment.json')['scene_extent'],
        cameras='Frozen matched-GT W2C/K; no camera optimization; both branches identical.',
        initialization='Identical untrained real-reference Omega surface 17112 plus 4000 low-opacity volume seeds; no photo-trained warmup. This is a shared real-derived geometry prior, not independent pure-eight-view initialization.',
        backend='3DGRUT model/optimizer/strategy with existing gsplat 3DGS rasterizer',
        loss='Existing tight loss: 0.8 foreground L1 + 0.2 SSIM + 0.2 outside opacity (3x3 dilation) + 0.03 opacity-weighted anisotropy above ratio16. Every image weight1.',
        sampling='Same 625 independently shuffled eight-image epochs; exactly 625 updates per image, 5000 total.',
        coverage_deg=[min(r['angle'] for r in a), max(r['angle'] for r in a)],
        limitations='Estimated real cameras; about39 degrees of supervision. 360-degree renders are extrapolation, not complete-car quality evidence. Masks differ by branch. Single reconstruction seed.',
        frozen_files=[dict(path=str(p), sha256=ab.digest(p)) for p in dict.fromkeys(frozen)]))
    print('PREPARED', OUT, flush=True)


def validate():
    for r in ab.read(OUT/'experiment.json')['frozen_files']:
        assert ab.digest(Path(r['path'])) == r['sha256'], r['path']
    for branch in ['real8','generated8']:
        for r in ab.read(OUT/branch/'inputs.json')['frames']:
            assert ab.digest(OUT/branch/'dataset/images'/r['name']) == r['sha256']
            assert ab.digest(OUT/branch/'dataset/images'/f"{Path(r['name']).stem}_mask.png") == r['mask_sha256']


def trainer_class():
    import torch
    import vggtdiff_car15_train as old
    from threedgrut.trainer import Trainer3DGRUT
    from threedgrut.datasets.utils import MultiEpochsDataLoader
    from threedgrut.model.losses import ssim
    from car15_object_loss import object_losses

    class PureEightTrainer(old.ObjectTrainer):
        def init_dataloaders(self, conf):
            Trainer3DGRUT.init_dataloaders(self, conf)
            names = [Path(p).name for p in self.train_dataset.image_paths]
            assert sorted(names) == [f'{i:02d}.png' for i in range(8)], names
            lookup = {int(Path(n).stem):i for i,n in enumerate(names)}
            rng = np.random.default_rng(SEED)
            stems = np.concatenate([rng.permutation(8) for _ in range(STEPS//8)])
            schedule = [lookup[int(i)] for i in stems]
            class FixedSampler(torch.utils.data.Sampler):
                def __iter__(self): return iter(schedule)
                def __len__(self): return len(schedule)
            self.train_dataloader = MultiEpochsDataLoader(self.train_dataset, batch_size=1, num_workers=0,
                sampler=FixedSampler(), pin_memory=False)
            self.sample_counts = Counter()
            ab.save(Path(conf.out_dir)/'sampling_plan.json', dict(stems=stems.tolist(), counts=dict(Counter(map(str,stems)))))

        def init_scene_extents(self, dataset):
            self.scene_extent = ab.read(OUT/'experiment.json')['scene_extent']
            bounds = ab.read(OUT/'shared/initialization.json')['bounds_world']
            self.scene_bbox = tuple(torch.tensor(v, device=self.device, dtype=torch.float32) for v in bounds)

        def setup_training(self, conf, model, train_dataset):
            Trainer3DGRUT.setup_training(self, conf, model, train_dataset)
            count = ab.read(OUT/'shared/initialization.json')['real_surface_points']
            with torch.no_grad():
                model.density[count:] = float(np.log(.01/.99))

        def get_losses(self, batch, outputs):
            self.sample_counts[Path(batch.image_name).name] += 1
            l1, struct, alpha = object_losses(outputs['pred_rgb'], batch.rgb_gt, outputs['pred_opacity'], batch.mask, ssim_fn=ssim, dilate=3)
            scales, opacity = self.model.get_scale(), self.model.get_density().reshape(-1)
            ratio = scales.amax(-1)/scales.amin(-1).clamp_min(1e-8)
            aniso = (torch.relu(torch.log(ratio)-math.log(16))*opacity).sum()/opacity.sum().clamp_min(1e-6)
            total = .8*l1+.2*struct+.2*alpha+.03*aniso
            zero = total.new_zeros(())
            return dict(total_loss=total,l1_loss=l1,l2_loss=zero,ssim_loss=struct,opacity_loss=alpha,scale_loss=aniso,lpips_loss=zero)
    return PureEightTrainer


def model_hash(model):
    h = hashlib.sha256()
    for key, val in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(branch):
    import torch
    from omegaconf import OmegaConf
    klass = trainer_class()
    from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR
    validate()
    root = OUT/branch
    if (root/'training/result.json').exists(): raise FileExistsError(branch)
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
    overrides = [f'path={root/"dataset"}', f'out_dir={root/"training"}', f'experiment_name={branch}',
        'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={OUT/"shared/initial_points.ply"}',
        'n_iterations=5000','num_workers=0','enable_writer=false','use_wandb=false','test_last=false',
        'compute_extra_metrics=false','export_ingp.enabled=false','export_ply.enabled=true',
        f'export_ply.path={root/"model.ply"}','dataset.test_split_interval=0','val_frequency=999999',
        'checkpoint.iterations=[1000,3000,5000]','model.progressive_training.max_n_features=1',
        'scheduler.positions.max_steps=5000','strategy.densify.end_iteration=3000',
        'strategy.prune.end_iteration=4500','strategy.reset_density.end_iteration=-1',f'+car15_stage={branch}']
    conf = compose_3dgrut_config('apps/colmap_3dgut',overrides,DEFAULT_THREEDGRUT_CONFIG_DIR)
    (root/'training').mkdir(parents=True,exist_ok=True)
    OmegaConf.save(conf,root/'training/config.yaml')
    start = time.time(); trainer = klass(conf); trainer.n_epochs = 1
    assert trainer.global_step == 0
    initial_hash = model_hash(trainer.model)
    ab.save(root/'training/initial_state.json',dict(sha256=initial_hash,gaussians=trainer.model.num_gaussians))
    print('TRAIN_START',branch,trainer.model.num_gaussians,initial_hash,flush=True)
    trainer.run_training(); trainer.save_checkpoint()
    assert trainer.global_step == STEPS and set(trainer.sample_counts.values()) == {625}
    checkpoint = Path(trainer.tracking.output_dir)/f'ours_{STEPS}'/f'ckpt_{STEPS}.pt'
    assert checkpoint.exists() and (root/'model.ply').exists()
    ab.save(root/'training/result.json',dict(branch=branch,step=trainer.global_step,checkpoint=str(checkpoint),
        elapsed_seconds=time.time()-start,gaussians=trainer.model.num_gaussians,initial_state_sha256=initial_hash,
        ply_sha256=ab.digest(root/'model.ply'),actual_sample_counts=dict(trainer.sample_counts)))
    print('TRAIN_COMPLETED',branch,flush=True)


def load(branch,step=STEPS):
    import torch
    from omegaconf import OmegaConf
    import vggtdiff_car15_train
    from threedgrut.model.model import MixtureOfGaussians
    root=OUT/branch
    checkpoint=Path(ab.read(root/'training/result.json')['checkpoint'])
    if step != STEPS: checkpoint=checkpoint.parent.parent/f'ours_{step}'/f'ckpt_{step}.pt'
    model=MixtureOfGaussians(OmegaConf.load(root/'training/config.yaml'))
    model.init_from_checkpoint(torch.load(checkpoint,map_location='cuda',weights_only=False),setup_optimizer=False)
    model.eval(); return model


def camera_sets():
    import vggtdiff_realvideo_unseen as geo
    rows=ab.read(OUT/'real8/inputs.json')['frames']
    c2w=np.linalg.inv(np.array([r['w2c'] for r in rows]))
    interpolator=Slerp(np.arange(8),Rotation.from_matrix(c2w[:,:3,:3]))
    local=[]
    for t in np.linspace(0,7,64):
        i=min(int(t),6); f=t-i
        c=np.eye(4); c[:3,:3]=interpolator(t).as_matrix(); c[:3,3]=(1-f)*c2w[i,:3,3]+f*c2w[i+1,:3,3]
        local.append(dict(w2c=np.linalg.inv(c).tolist(),k=rows[0]['k'],angle=(1-f)*rows[i]['angle']+f*rows[i+1]['angle']))
    g=ab.read(ab.BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json')
    center,up,front,left=[np.array(g[k]) for k in ['center','up','front','left']]
    positions=c2w[:,:3,3]-center
    heights=positions@up
    radius=float(np.median(np.linalg.norm(positions-heights[:,None]*up,axis=1)))
    height=float(np.median(heights))
    orbit=[]
    for a in np.arange(64)*360/64:
        th=np.deg2rad(a)
        p=center+radius*(np.cos(th)*front-np.sin(th)*left)+height*up
        orbit.append(dict(angle=float(a),w2c=geo.look_at(p,center,up).tolist(),k=rows[0]['k']))
    elevated=[]
    for r in local:
        c=np.linalg.inv(r['w2c']); c[:3,3]+=.025*up
        # Preserve orientation: this tests a small vertical camera displacement.
        elevated.append(dict(r,w2c=np.linalg.inv(c).tolist()))
    return dict(local=local,elevated=elevated,orbit360=orbit)


def score(rgb,alpha,row):
    from vggtdiff_car15_evaluate import metrics
    gt=np.asarray(Image.open(row['image']).convert('RGB'),dtype=float)/255
    mask=np.asarray(Image.open(row['mask']).convert('L'))>127
    result=metrics(rgb,alpha,gt,mask)
    for threshold in [.05,.5]:
        pred=alpha>threshold
        result[f'outside_area_over_gt_at_{threshold}']=float((pred&~mask).sum()/mask.sum())
        result[f'missing_gt_fraction_at_{threshold}']=float((mask&~pred).sum()/mask.sum())
    result['outside_opacity_over_gt_area']=float(alpha[~mask].sum()/mask.sum())
    return result


def evaluate():
    import torch
    from plyfile import PlyData
    from omegaconf import OmegaConf
    import vggtdiff_car15_train  # Install the rendering backend and vendor import path.
    from vggtdiff_selfconsistent_silhouette import render
    from threedgrut.model.model import MixtureOfGaussians
    validate(); torch.set_num_threads(4)
    sets=camera_sets(); ab.save(OUT/'evaluation/cameras.json',sets)
    real=ab.read(OUT/'real8/inputs.json')['frames']
    refs=ab.read(MATCH/'references.json')['frames']
    refs=[dict(r,k=r['K'],w2c=r['T'],angle=r['azimuth_deg']) for r in refs]
    # These two real references lie inside the eight-view supervision arc.
    heldout=[r for r in refs if real[0]['angle'] <= r['angle'] <= real[-1]['angle']]
    assert [r['original_frame'] for r in heldout] == [960,1200]
    report={}
    for branch in ['real8','generated8']:
        model=load(branch); root=OUT/branch/'evaluation'; own=ab.read(OUT/branch/'inputs.json')['frames']
        scores={}
        for category,rows in [('own_training',own),('common_real_at_training_cameras',real),('heldout_real_in_arc',heldout)]:
            values=[]
            for i,r in enumerate(rows):
                rgb,alpha,depth=render(model,r)
                assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all()
                values.append(dict(frame=r['original_frame'],**score(rgb,alpha,r)))
                ab.write_rgb(root/category/f'{i:03d}.png',rgb)
                gt=np.asarray(Image.open(r['image']).convert('RGB'),dtype=float)/255
                mask=np.asarray(Image.open(r['mask']).convert('L'))>127
                ab.write_rgb(OUT/'evaluation/groundtruth'/category/f'{i:03d}.png',gt*mask[:,:,None])
            scores[category]=dict(mean={k:float(np.mean([v[k] for v in values])) for k in values[0] if k!='frame'},frames=values)
        for category,rows in sets.items():
            for i,r in enumerate(rows):
                rgb,alpha,depth=render(model,r)
                assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all()
                ab.write_rgb(root/category/f'{i:03d}.png',rgb)
                if category in ['local','elevated']:
                    ab.write_rgb(root/(category+'_alpha')/f'{i:03d}.png',np.repeat(alpha[:,:,None],3,axis=2))
            print('RENDERED',branch,category,flush=True)
        ply_model=MixtureOfGaussians(OmegaConf.load(OUT/branch/'training/config.yaml'))
        ply_model.init_from_ply(str(OUT/branch/'model.ply'),init_model=False); ply_model.eval()
        errors=[]
        for i in [0,16,32,48]:
            a,_,_=render(model,sets['orbit360'][i]); b,_,_=render(ply_model,sets['orbit360'][i]); errors.append(float(abs(a-b).mean()))
        assert max(errors)<1e-6,errors
        p=PlyData.read(OUT/branch/'model.ply')['vertex'].data
        assert all(np.isfinite(p[k]).all() for k in p.dtype.names)
        scales=np.exp(np.column_stack([p[f'scale_{i}'] for i in range(3)]).astype(float)); scales.sort(axis=1)
        ratios=scales[:,-1]/np.maximum(scales[:,0],1e-12)
        opacity=1/(1+np.exp(-p['opacity'].astype(float)))
        scores['model']=dict(gaussians=len(p),ratio_quantiles=np.quantile(ratios,[.5,.9,.99]).tolist(),
            opacity_fraction_ratio_above16=float(opacity[ratios>16].sum()/opacity.sum()),ply_roundtrip_rgb_mae=errors)
        report[branch]=scores
        del model,ply_model; torch.cuda.empty_cache()
        ab.save(OUT/'evaluation/summary.json',report)
    initials=[ab.read(OUT/b/'training/initial_state.json') for b in ['real8','generated8']]
    assert initials[0]==initials[1],initials
    for b in ['real8','generated8']:
        assert set(ab.read(OUT/b/'training/result.json')['actual_sample_counts'].values())=={625}
    ab.save(OUT/'evaluation/audit.json',dict(state='passed',identical_initial_model=initials[0],
        exact_camera_match=True,every_image_updates=625,total_updates_per_branch=5000,ply_roundtrip=True,finite_ply_and_renders=True))
    print('EVALUATION_COMPLETED',flush=True)


def compare():
    for category in ['local','elevated','orbit360','local_alpha','elevated_alpha','own_training','common_real_at_training_cameras','heldout_real_in_arc']:
        paths=sorted((OUT/'real8/evaluation'/category).glob('*.png'))
        if not paths: raise FileNotFoundError(category)
        dest=OUT/'evaluation'/category
        for i,path in enumerate(paths):
            canvas=Image.new('RGB',(1664,512),'#181818'); draw=ImageDraw.Draw(canvas)
            for j,b in enumerate(['real8','generated8']):
                canvas.paste(Image.open(OUT/b/'evaluation'/category/path.name),(j*832,32))
                draw.text((j*832+10,10),f'{b} | {category} | {i:02d}',fill='white')
            dest.mkdir(parents=True,exist_ok=True); canvas.save(dest/path.name)
        if len(paths)==64:
            subprocess.run(['ffmpeg','-y','-framerate','12','-i',str(dest/'%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(OUT/'evaluation'/f'{category}_comparison.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        indices=np.linspace(0,len(paths)-1,min(8,len(paths))).astype(int)
        ab.montage([dest/paths[i].name for i in indices],[f'{category} {i:02d}' for i in indices],OUT/'evaluation'/f'{category}_contact.jpg',2,(832,256))
    print('COMPARISON_READY',OUT/'evaluation',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','train','evaluate','compare']); p.add_argument('--branch',choices=['real8','generated8'])
    args=p.parse_args()
    if args.action=='train':
        if args.branch is None: p.error('train requires --branch')
        train(args.branch)
    else: globals()[args.action]()

if __name__=='__main__': main()
