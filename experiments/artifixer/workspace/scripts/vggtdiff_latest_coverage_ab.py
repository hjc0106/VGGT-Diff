"""Controlled coverage experiment on the latest 64-view white-car generation.

Prepare -> inspect masks -> approve-mask-review -> train A/B -> evaluate -> compare.
All outputs are isolated; existing experiments and original images stay untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from vggtdiff_white_car import BASE, REPO, digest, save

SOURCE = BASE / 'vggtdiff_realvideo_orbit360_64_latest_refs_v1'
OLD = BASE / 'vggtdiff_car15_reconstruction_v1'
OUT = BASE / 'vggtdiff_latest_coverage_ab_v1'
RIGHT = [8, 10, 12, 14, 16, 18, 20, 21, 23, 24, 27]
ADDED = [0, 3, 30, 32, 33, 61]
SEED = 20261021
START, END = 1000, 5000


def read(path):
    return json.loads(Path(path).read_text())


def frozen_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    if digest(source) != digest(destination):
        raise ValueError(f'Frozen copy differs: {destination}')


def montage(paths, labels, destination, columns=3, tile=(416, 240)):
    w, h = tile
    canvas = Image.new('RGB', (columns*w, math.ceil(len(paths)/columns)*(h+28)), '#181818')
    draw = ImageDraw.Draw(canvas)
    for j, (path, label) in enumerate(zip(paths, labels)):
        x, y = j % columns*w, j // columns*(h+28)
        with Image.open(path) as image:
            canvas.paste(image.convert('RGB').resize(tile, Image.Resampling.LANCZOS), (x, y+28))
        draw.text((x+6, y+7), label, fill='white')
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95)


def write_colmap(root, rows):
    images, sparse = root/'images', root/'sparse/0'
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    camera_lines, image_lines = [], []
    with (sparse/'cameras.bin').open('wb') as cameras, (sparse/'images.bin').open('wb') as records:
        cameras.write(struct.pack('<Q', len(rows)))
        records.write(struct.pack('<Q', len(rows)))
        for cid, row in enumerate(rows, 1):
            k, w = np.array(row['k']), np.array(row['w2c'])
            quaternion = Rotation.from_matrix(w[:3, :3]).as_quat()
            q = np.r_[quaternion[3], quaternion[:3]]
            params = [k[0, 0], k[1, 1], k[0, 2], k[1, 2], 0, 0, 0, 0]
            cameras.write(struct.pack('<iiQQ8d', cid, 4, 832, 480, *params))
            records.write(struct.pack('<i7di', cid, *q, *w[:3, 3], cid))
            records.write(row['name'].encode()+b'\0')
            records.write(struct.pack('<Q', 0))
            camera_lines.append(' '.join(map(str, [cid, 'OPENCV', 832, 480, *params])))
            image_lines.append(' '.join(map(str, [cid, *q, *w[:3, 3], cid, row['name']]))+'\n')
            frozen_copy(Path(row['image']), images/row['name'])
            frozen_copy(Path(row['mask']), images/f"{Path(row['name']).stem}_mask.png")
    (sparse/'cameras.txt').write_text('\n'.join(camera_lines)+'\n')
    (sparse/'images.txt').write_text('\n'.join(image_lines)+'\n')
    (sparse/'points3D.txt').write_text('# Explicit common fused initialization\n')
    frozen_copy(OUT/'shared/initial_points.ply', root/'point_cloud.ply')


def prepare():
    import torch
    from vggtdiff_car15_groupa_prepare import choose_mask
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter

    torch.set_num_threads(4)
    if read(SOURCE/'status.json')['state'] != 'completed':
        raise ValueError('Source generation is incomplete')
    trajectory = read(SOURCE/'trajectory.json')
    records = read(SOURCE/'result.json')['frames']
    if [r['index'] for r in records] != list(range(64)):
        raise ValueError('Expected exactly 64 ordered generated views')
    for name in ['initial_points.ply', 'initialization.json', 'warmup_checkpoint.pt']:
        frozen_copy(OLD/name, OUT/'shared'/name)
    real = read(OLD/'inputs.json')['frames'][:6]
    segmenter = None
    mask_stats, all_generated = [], []
    for record in records:
        i = record['index']
        source = Path(record['path'])
        if digest(source) != record['sha256']:
            raise ValueError(f'Source hash mismatch: {source}')
        image = np.asarray(Image.open(source).convert('RGB'))
        if image.shape != (480, 832, 3):
            raise ValueError(f'Unexpected source shape: {source}')
        mask_path = OUT/'pool/masks'/f'{i:02d}.png'
        if not mask_path.exists():
            if segmenter is None:
                segmenter = SAM3Segmenter(SimpleNamespace(
                    source_dir=str(REPO/'thirdparty/sam3'),
                    checkpoint_path=str(REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
                    device='cuda', confidence_threshold=.15, instance_strategy='best_score'))
            mask = choose_mask(segmenter, image)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype('uint8')*255).save(mask_path)
        mask = np.asarray(Image.open(mask_path).convert('L')) > 127
        ys, xs = np.where(mask)
        if mask.shape != (480, 832) or len(xs) < 1000:
            raise ValueError(f'Bad mask: {mask_path}')
        overlay = image.copy()
        overlay[mask] = (.6*image[mask]+np.array([0, 255, 0])*.4).astype('uint8')
        preview = OUT/'pool/overlays'/f'{i:02d}.png'
        preview.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(preview)
        rgba = OUT/'pool/rgba'/f'{i:02d}.png'
        rgba.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.dstack([image, mask.astype('uint8')*255])).save(rgba)
        w, k = np.array(trajectory['target_w2c'][i]), np.array(trajectory['target_intrinsics'][i])
        if not np.allclose(w[:3, :3]@w[:3, :3].T, np.eye(3), atol=1e-6):
            raise ValueError('Invalid rotation')
        row = dict(kind='generated', orbit_index=i, angle=float(trajectory['target_angles_deg'][i]),
                   image=str(source), mask=str(mask_path), sha256=digest(source), mask_sha256=digest(mask_path),
                   w2c=w.tolist(), k=k.tolist())
        all_generated.append(row)
        mask_stats.append(dict(orbit_index=i, area=float(mask.mean()),
                               bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                               sha256=digest(mask_path)))
        print('MASK', i, flush=True)
    del segmenter
    torch.cuda.empty_cache()
    all_real = []
    for i, source in enumerate(real):
        image, mask = OLD/'images'/source['name'], OLD/'masks'/source['name']
        if digest(image) != source['sha256']:
            raise ValueError('Real image changed')
        all_real.append(dict(source, name=f'{i:02d}.png', image=str(image), mask=str(mask), mask_sha256=digest(mask)))
    for branch, indices in [('A', RIGHT), ('B', RIGHT+ADDED)]:
        rows = [dict(r) for r in all_real]
        rows += [dict(all_generated[i], name=f'{j+6:02d}.png') for j, i in enumerate(indices)]
        write_colmap(OUT/branch/'dataset', rows)
        save(OUT/branch/'inputs.json', dict(frames=rows, generated_indices=indices,
                                          coordinates='Original VGGT-Omega world; OpenCV W2C; unmodified 832x480'))
    # Use one frozen extent in both branches: data coverage must not change splitting thresholds.
    old_rows = read(OLD/'inputs.json')['frames']
    centers = np.linalg.inv(np.array([r['w2c'] for r in old_rows]))[:, :3, 3]
    scene_extent = float(np.linalg.norm(centers-centers.mean(0), axis=1).max()*1.1)
    files = [SOURCE/'trajectory.json', SOURCE/'result.json', OLD/'validation/cameras.json',
             OUT/'shared/initial_points.ply', OUT/'shared/warmup_checkpoint.pt',
             Path(__file__), REPO/'scripts/car15_gsplat_backend.py', REPO/'scripts/car15_object_loss.py',
             REPO/'scripts/vggtdiff_car15_train.py', REPO/'scripts/vggtdiff_car15_groupa_tight_train.py']
    files += [Path(r['image']) for r in all_real+all_generated]+[Path(r['mask']) for r in all_real+all_generated]
    for row in read(OLD/'validation/cameras.json')['frames']:
        files.extend([OLD/'validation/images'/row['name'], OLD/'validation/masks'/row['name']])
    save(OUT/'pool.json', dict(real=all_real, generated=all_generated, mask_statistics=mask_stats))
    save(OUT/'experiment.json', dict(seed=SEED, start_step=START, final_step=END, scene_extent=scene_extent,
        A_generated_indices=RIGHT, B_added_indices=ADDED, generated_loss_weight=.3,
        backend='3DGRUT optimizer/model/strategy + gsplat 3DGS rasterizer',
        loss='Existing tight loss: anisotropy .03 above ratio 16; outside opacity .2; 3x3 mask dilation',
        sampling='Same deterministic real/generated role and real-image sequence; per-branch uniform generated draws',
        interpretation='Same total update budget; adding generated views reduces mean exposure per generated image',
        frozen_files=[dict(path=str(f), sha256=digest(f)) for f in files]))
    montage([OUT/'pool/overlays'/f'{i:02d}.png' for i in range(64)],
            [f'{i:02d} | {i*5.625:g} deg' for i in range(64)], OUT/'masks_all64.jpg', 8, (208, 120))
    for name, indices in [('right11', RIGHT), ('added6', ADDED)]:
        montage([OUT/'pool/overlays'/f'{i:02d}.png' for i in indices],
                [f'{i:02d} | {i*5.625:g} deg' for i in indices], OUT/f'masks_{name}.jpg')
    save(OUT/'status.json', dict(state='prepared_pending_mask_review', A_images=17, B_images=23))
    print('PREPARED', OUT, flush=True)


def validate_frozen():
    for row in read(OUT/'experiment.json')['frozen_files']:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f"Frozen input changed: {row['path']}")


def review_masks():
    validate_frozen()
    save(OUT/'mask_review.json', dict(state='accepted', training_indices=RIGHT+ADDED,
        excluded_training_indices=[34, 35, 36, 37],
        notes='Assistant visually inspected added6, right11 and all64 overlays before training. '
              'Car body/wheels/mirrors retained; distant vehicles, road and dark block artifacts excluded. '
              'Silhouette and generated geometry remain approximate.'))


def training_components():
    import torch
    import vggtdiff_car15_train  # Adds the 3DGRUT import path and installs the gsplat adapter.
    from vggtdiff_car15_groupa_tight_train import TightTrainer
    from threedgrut.trainer import Trainer3DGRUT
    from threedgrut.datasets.utils import MultiEpochsDataLoader

    class CoverageTrainer(TightTrainer):
        def init_dataloaders(self, conf):
            Trainer3DGRUT.init_dataloaders(self, conf)
            names = [Path(p).name for p in self.train_dataset.image_paths]
            count = len(names)-6
            lookup = {int(Path(name).stem): i for i, name in enumerate(names)}
            rng = np.random.default_rng(SEED)
            roles = rng.random(END-START) < .5
            real_indices = rng.integers(0, 6, END-START)
            generated_uniform = rng.random(END-START)
            stems = np.where(roles, real_indices, 6+(generated_uniform*count).astype(int))
            schedule = [lookup[int(i)] for i in stems]
            class FixedSampler(torch.utils.data.Sampler):
                def __iter__(self): return iter(schedule)
                def __len__(self): return len(schedule)
            self.train_dataloader = MultiEpochsDataLoader(self.train_dataset, batch_size=1,
                num_workers=0, sampler=FixedSampler(), pin_memory=False)
            self.sample_counts = Counter()
            save(Path(conf.out_dir)/f'{conf.car15_stage}_sampling_plan.json',
                 dict(names=names, stems=stems.tolist(), real_updates=int(roles.sum()),
                      generated_updates=int((~roles).sum()), counts=dict(Counter(map(str, stems)))))

        def init_scene_extents(self, dataset):
            self.scene_extent = read(OUT/'experiment.json')['scene_extent']
            bounds = read(OUT/'shared/initialization.json')['bounds_world']
            self.scene_bbox = tuple(torch.tensor(x, device=self.device, dtype=torch.float32) for x in bounds)

        def get_losses(self, batch, outputs):
            self.sample_counts[Path(batch.image_name).name] += 1
            return super().get_losses(batch, outputs)

    return CoverageTrainer


def train(branch):
    import torch
    from omegaconf import OmegaConf
    trainer_class = training_components()
    from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR
    validate_frozen()
    if read(OUT/'mask_review.json')['state'] != 'accepted':
        raise ValueError('Inspect masks before training')
    root = OUT/branch
    if (root/'training/result.json').exists():
        raise FileExistsError(f'{branch} already trained')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    overrides = [f'path={root/"dataset"}', f'out_dir={root/"training"}', f'experiment_name={branch}',
        'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={OUT/"shared/initial_points.ply"}',
        'n_iterations=5000', 'num_workers=0', 'enable_writer=false', 'use_wandb=false',
        'test_last=false', 'compute_extra_metrics=false', 'export_ingp.enabled=false', 'export_ply.enabled=true',
        f'export_ply.path={root/"model.ply"}', 'dataset.test_split_interval=0', 'val_frequency=999999',
        'checkpoint.iterations=[3000,5000]', 'model.progressive_training.max_n_features=1',
        'scheduler.positions.max_steps=5000', 'strategy.densify.end_iteration=3000', 'strategy.prune.end_iteration=4500',
        'strategy.reset_density.end_iteration=-1', f'+car15_stage={branch}', f'resume={OUT/"shared/warmup_checkpoint.pt"}']
    conf = compose_3dgrut_config('apps/colmap_3dgut', overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)
    (root/'training').mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, root/'training/config.yaml')
    start = time.time()
    trainer = trainer_class(conf)
    if trainer.global_step != START:
        raise ValueError('Wrong warmup step')
    trainer.n_epochs = 1  # One fixed 4000-update schedule, independent of dataset length.
    save(root/'status.json', dict(state='training', start_step=START, final_step=END))
    print('TRAIN_START', branch, trainer.global_step, trainer.model.num_gaussians, flush=True)
    trainer.run_training()
    trainer.save_checkpoint()
    assert trainer.global_step == END and sum(trainer.sample_counts.values()) == END-START
    checkpoint = Path(trainer.tracking.output_dir)/f'ours_{END}'/f'ckpt_{END}.pt'
    assert checkpoint.exists()
    save(root/'training/result.json', dict(branch=branch, step=trainer.global_step,
        gaussians=trainer.model.num_gaussians, elapsed_seconds=time.time()-start,
        checkpoint=str(checkpoint), ply=str(root/'model.ply'), ply_sha256=digest(root/'model.ply'),
        warmup_sha256=digest(OUT/'shared/warmup_checkpoint.pt'), scene_extent=trainer.scene_extent,
        actual_sample_counts=dict(trainer.sample_counts)))
    save(root/'status.json', dict(state='trained_pending_evaluation', step=END))
    print('TRAIN_COMPLETED', branch, trainer.global_step, trainer.model.num_gaussians, flush=True)


def load(branch, step=END):
    import torch
    from omegaconf import OmegaConf
    import vggtdiff_car15_train  # Installs the same gsplat backend used for training.
    from threedgrut.model.model import MixtureOfGaussians
    root = OUT/branch
    result = read(root/'training/result.json')
    checkpoint = Path(result['checkpoint'])
    if step != END:
        checkpoint = checkpoint.parent.parent/f'ours_{step}'/f'ckpt_{step}.pt'
    model = MixtureOfGaussians(OmegaConf.load(root/'training/config.yaml'))
    model.init_from_checkpoint(torch.load(checkpoint, map_location='cuda', weights_only=False), setup_optimizer=False)
    model.eval()
    return model


def mean_metrics(rows):
    keys = ['foreground_psnr', 'foreground_mae', 'silhouette_iou', 'foreground_mean_opacity']
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} if rows else {}


def write_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1)*255).astype('uint8')).save(path)


def evaluate(branch):
    import torch
    from plyfile import PlyData
    from vggtdiff_car15_evaluate import render, metrics
    import vggtdiff_controlled_context as cameras
    import vggtdiff_realvideo_unseen as geometry

    validate_frozen()
    torch.set_num_threads(4)
    root = OUT/branch/'evaluation'
    model = load(branch)
    pool = read(OUT/'pool.json')
    selected = read(OUT/branch/'inputs.json')['generated_indices']
    real_rows, generated_rows, validation_rows = [], [], []

    def measure(row, image, mask):
        rgb, alpha = render(model, np.array(row['w2c']), row['k'])
        gt = np.asarray(Image.open(image).convert('RGB'), dtype=float)/255
        mask = np.asarray(Image.open(mask).convert('L')) > 127
        return rgb, alpha, metrics(rgb, alpha, gt, mask)

    for row in pool['real']:
        rgb, alpha, score = measure(row, row['image'], row['mask'])
        real_rows.append(dict(name=row['name'], **score))
        write_rgb(root/'real_training'/row['name'], rgb)
    for row in pool['generated']:
        i = row['orbit_index']
        rgb, alpha, score = measure(row, row['image'], row['mask'])
        generated_rows.append(dict(index=i, angle=row['angle'], in_training=i in selected, **score))
        write_rgb(root/'orbit'/f'{i:03d}.png', rgb)
        (root/'opacity').mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(alpha, 0, 1)*255).astype('uint8')).save(root/'opacity'/f'{i:03d}.png')
    for row in read(OLD/'validation/cameras.json')['frames']:
        name = row['name']
        rgb, alpha, score = measure(row, OLD/'validation/images'/name, OLD/'validation/masks'/name)
        validation_rows.append(dict(name=name, **score))
        write_rgb(root/'validation'/name, rgb)
    trajectory = read(SOURCE/'trajectory.json')
    offset = cameras.trajectory([i*5.625+2.8125 for i in range(64)])
    assert np.allclose(cameras.trajectory([i*5.625 for i in range(64)])['target_w2c'], trajectory['target_w2c'])
    for i, (w, k) in enumerate(zip(offset['target_w2c'], offset['target_intrinsics'])):
        rgb, alpha = render(model, np.array(w), k)
        write_rgb(root/'offset'/f'{i:03d}.png', rgb)
    g = read(BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json')
    center, up, front, left = [np.array(g[k]) for k in ['center', 'up', 'front', 'left']]
    elevated = []
    for angle in range(0, 360, 60):
        theta = np.deg2rad(angle)
        position = center+.3142*(np.cos(theta)*front-np.sin(theta)*left)+.10749*up
        w = geometry.look_at(position, center, up)
        rgb, alpha = render(model, w, trajectory['target_intrinsics'][0])
        write_rgb(root/'elevated'/f'{angle:03d}.png', rgb)
        elevated.append(dict(angle=angle, w2c=w.tolist(), k=trajectory['target_intrinsics'][0]))
    # Verify PLY export preserves actual model rendering, not only finite parameters.
    from omegaconf import OmegaConf
    from threedgrut.model.model import MixtureOfGaussians
    ply_model = MixtureOfGaussians(OmegaConf.load(OUT/branch/'training/config.yaml'))
    ply_model.init_from_ply(str(OUT/branch/'model.ply'), init_model=False)
    ply_model.eval()
    roundtrip = []
    for i in [0, 16, 32, 48]:
        w, k = trajectory['target_w2c'][i], trajectory['target_intrinsics'][i]
        ref, _ = render(model, np.array(w), k)
        check, _ = render(ply_model, np.array(w), k)
        roundtrip.append(float(np.abs(ref-check).mean()))
    assert max(roundtrip) < .001, roundtrip
    del ply_model, model
    torch.cuda.empty_cache()
    # A small step-3000 checkpoint comparison detects late fitting regressions.
    interim = load(branch, 3000)
    interim_validation = []
    for row in read(OLD/'validation/cameras.json')['frames']:
        rgb, alpha = render(interim, np.array(row['w2c']), row['k'])
        gt = np.asarray(Image.open(OLD/'validation/images'/row['name']).convert('RGB'), dtype=float)/255
        mask = np.asarray(Image.open(OLD/'validation/masks'/row['name']).convert('L')) > 127
        interim_validation.append(dict(name=row['name'], **metrics(rgb, alpha, gt, mask)))
    for i in [0, 16, 24, 32]:
        rgb, _ = render(interim, np.array(trajectory['target_w2c'][i]), trajectory['target_intrinsics'][i])
        write_rgb(root/'step3000'/f'{i:03d}.png', rgb)
    del interim
    torch.cuda.empty_cache()
    ply = PlyData.read(OUT/branch/'model.ply')['vertex'].data
    assert all(np.isfinite(ply[k]).all() for k in ply.dtype.names)
    scale = np.exp(np.column_stack([ply[f'scale_{i}'] for i in range(3)]).astype(float))
    scale.sort(axis=1)
    opacity = 1/(1+np.exp(-ply['opacity'].astype(float)))
    diagnostics = {}
    for key, ratio in [('longest_over_shortest', scale[:, 2]/np.maximum(scale[:, 0], 1e-12)),
                       ('needle_longest_over_middle', scale[:, 2]/np.maximum(scale[:, 1], 1e-12)),
                       ('flat_middle_over_shortest', scale[:, 1]/np.maximum(scale[:, 0], 1e-12))]:
        diagnostics[key] = dict(quantiles=np.quantile(ratio, [.5, .9, .99]).tolist(),
                               opacity_sum_fraction_above16=float(opacity[ratio > 16].sum()/opacity.sum()))
    report = dict(branch=branch, gaussians=len(ply), real_training=mean_metrics(real_rows),
        generated_training=mean_metrics([r for r in generated_rows if r['in_training']]),
        generated_holdout=mean_metrics([r for r in generated_rows if not r['in_training']]),
        common_right11=mean_metrics([r for r in generated_rows if r['index'] in RIGHT]),
        added_front_rear6=mean_metrics([r for r in generated_rows if r['index'] in ADDED]),
        common_generated_holdout=mean_metrics([r for r in generated_rows if r['index'] not in RIGHT+ADDED]),
        independent_real=mean_metrics(validation_rows), step3000_independent_real=mean_metrics(interim_validation),
        real_rows=real_rows, generated_rows=generated_rows, validation_rows=validation_rows,
        gaussian_diagnostics=diagnostics, ply_roundtrip_mae=roundtrip,
        caution='Generated holdout is same-generation consistency, not independent physical geometry truth; '
                'real validation has estimated PnP poses and observes only the recorded side.')
    save(root/'metrics.json', report)
    save(root/'novel_cameras.json', dict(offset=offset, elevated=elevated))
    save(OUT/branch/'status.json', dict(state='evaluated_pending_visual_review', step=END))
    print('EVALUATED', branch, {k: report[k] for k in ['real_training', 'independent_real', 'added_front_rear6']}, flush=True)


def video(folder, output):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '8', '-i', str(folder/'%03d.png'),
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(output)], check=True)


def compare():
    from vggtdiff_car15_keep6 import pair
    validate_frozen()
    root = OUT/'comparison'
    for category, indices in [('orbit', range(64)), ('offset', range(64)), ('elevated', range(0, 360, 60))]:
        for i in indices:
            angle = i*5.625+(2.8125 if category == 'offset' else 0) if category != 'elevated' else i
            pair(OUT/'A/evaluation'/category/f'{i:03d}.png', OUT/'B/evaluation'/category/f'{i:03d}.png',
                 root/category/f'{i:03d}.png', f'A right11 | {angle:g} deg', f'B plus front/rear | {angle:g} deg')
        if category != 'elevated':
            video(root/category, root/f'{category}_A_vs_B.mp4')
    selected = [0, 3, 8, 16, 24, 30, 32, 33, 48, 61]
    montage([root/'orbit'/f'{i:03d}.png' for i in selected], [f'{i*5.625:g} deg' for i in selected],
            root/'key_views.jpg', 2, (832, 256))
    montage([root/'orbit'/f'{i:03d}.png' for i in range(64)], [f'{i*5.625:g} deg' for i in range(64)],
            root/'all64.jpg', 4, (416, 128))
    montage([root/'elevated'/f'{i:03d}.png' for i in range(0, 360, 60)], [f'{i} deg' for i in range(0, 360, 60)],
            root/'elevated.jpg', 2, (832, 256))
    for name in ['600.png', '1080.png']:
        canvas = Image.new('RGB', (832*3, 512), '#181818')
        for j, (path, label) in enumerate([(OLD/'validation/images'/name, 'Independent real'),
            (OUT/'A/evaluation/validation'/name, 'A right11'), (OUT/'B/evaluation/validation'/name, 'B plus front/rear')]):
            canvas.paste(Image.open(path), (j*832, 32))
            ImageDraw.Draw(canvas).text((j*832+10, 10), label, fill='white')
        canvas.save(root/f'validation_{name}')
    a, b = [read(OUT/branch/'evaluation/metrics.json') for branch in ['A', 'B']]
    results = {branch: read(OUT/branch/'training/result.json') for branch in ['A', 'B']}
    assert results['A']['warmup_sha256'] == results['B']['warmup_sha256']
    for i in range(6):
        assert results['A']['actual_sample_counts'].get(f'{i:02d}.png', 0) == results['B']['actual_sample_counts'].get(f'{i:02d}.png', 0)
    rows = ['# 最新 64 帧生成图：覆盖范围 A/B 重建实验', '',
        'A：六张真实图＋右侧十一张最新生成图；B：相同数据再加入六张正前/正后附近图，共二十三张。', '',
        '相同的 1000 步真实图检查点、初始化、固定相机、场景尺度、紧约束损失；各训练 4000 次更新至 5000 步。',
        '真实/生成采样角色及每张真实图曝光次数完全一致；B 的生成图更多，每张生成图的平均曝光次数降低。', '',
        '| 评估集合 | A PSNR | B PSNR | A IoU | B IoU |', '|---|---:|---:|---:|---:|']
    for key in ['real_training', 'independent_real', 'common_right11', 'added_front_rear6', 'common_generated_holdout']:
        rows.append(f"| {key} | {a[key]['foreground_psnr']:.3f} | {b[key]['foreground_psnr']:.3f} | "
                    f"{a[key]['silhouette_iou']:.4f} | {b[key]['silhouette_iou']:.4f} |")
    rows += ['', '生成图指标仅衡量合成监督一致性，不能当作未见侧真实几何精度。真实验证相机来自 PnP，存在估计误差。', '',
        '已输出整圈、半步偏移环绕、六个抬高角度及两张独立真实验证图；仍需视觉检查后判断是否进入实验 C。', '',
        '## 主要产物', '', '- A/model.ply、B/model.ply；各自 training 中的 3000/5000 步检查点。',
        '- comparison/orbit_A_vs_B.mp4、comparison/offset_A_vs_B.mp4。',
        '- comparison/key_views.jpg、comparison/elevated.jpg、comparison/validation_600.png 和 validation_1080.png。',
        '- A/evaluation/metrics.json、B/evaluation/metrics.json，含逐帧数据和高斯形状诊断。',
        '- experiment.json 记录冻结哈希与配置，mask_review.json 记录分割检查。', '',
        '## 复现', '', '使用 .venv-car15/bin/python scripts/vggtdiff_latest_coverage_ab.py：',
        'prepare → approve-mask-review（检查分割图后）→ train --branch A/B → evaluate --branch A/B → compare。', '']
    (OUT/'REPORT.md').write_text('\n'.join(rows))
    save(OUT/'status.json', dict(state='compared_pending_visual_review', A_images=17, B_images=23,
                               shared_warmup_verified=True, matching_real_sampling_verified=True))
    print('COMPARISON_COMPLETED', OUT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'approve-mask-review', 'train', 'evaluate', 'compare'])
    parser.add_argument('--branch', choices=['A', 'B'])
    args = parser.parse_args()
    if args.action in ['train', 'evaluate']:
        if args.branch is None:
            parser.error('--branch is required')
        globals()[args.action](args.branch)
    elif args.action == 'approve-mask-review':
        review_masks()
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
