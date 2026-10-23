"""Retrain with the tight loss plus the five rear-right generated views.

Six real images and the six generated views from 45° to 101.25° stay as they
were in keep6. The new supervision is the joint batch at 112.5°, 120°,
127.5°, 135° and 150°. Generated loss weight remains 0.3. Each group is
sampled with equal total probability.
"""
import shutil
import struct
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation

import vggtdiff_car15_keep6 as keep
from vggtdiff_car15_groupa_prepare import OUT as GROUP
from vggtdiff_car15_groupa_tight_train import ANISO_WEIGHT, OUTSIDE_WEIGHT, RATIO_CAP, TightTrainer
from vggtdiff_white_car import BASE, digest, save
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.datasets.utils import MultiEpochsDataLoader
from threedgrut.model.model import MixtureOfGaussians
from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR

OUT = BASE / 'vggtdiff_car15_rear5_v1'
REAR = BASE / 'vggtdiff_realvideo_rear_iter1_v1'
PRIOR_ANGLES = [45.0, 56.25, 67.5, 78.75, 90.0, 101.25]
REAR_ANGLES = [112.5, 120.0, 127.5, 135.0, 150.0]
REAR_FILES = {
    112.5: ('raw/00.png', 'masks/04_new_112.5.png'),
    120.0: ('raw/01.png', 'masks/05_new_120.png'),
    127.5: ('raw/02.png', 'masks/06_new_127.5.png'),
    135.0: ('raw/03.png', 'masks/07_new_135.png'),
    150.0: ('raw/04.png', 'masks/08_new_150.png'),
}


def read(path):
    import json
    return json.loads(Path(path).read_text())


def write_colmap(root, rows):
    images = root / 'images'
    sparse = root / 'sparse/0'
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    cameras, records = [], []
    for camera_id, row in enumerate(rows, start=1):
        intrinsics = np.array(row['k'])
        pose = np.array(row['w2c'])
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
        quaternion = np.r_[quaternion[3], quaternion[:3]]
        cameras.append(f'{camera_id} OPENCV 832 480 {intrinsics[0, 0]:.17g} {intrinsics[1, 1]:.17g} {intrinsics[0, 2]:.17g} {intrinsics[1, 2]:.17g} 0 0 0 0')
        records.append(' '.join(map(str, [camera_id, *quaternion, *pose[:3, 3], camera_id, row['name']])))
        shutil.copyfile(OUT / 'images' / row['name'], images / row['name'])
        shutil.copyfile(OUT / 'masks' / row['name'], images / f'{Path(row["name"]).stem}_mask.png')
    (sparse / 'cameras.txt').write_text('\n'.join(cameras) + '\n')
    (sparse / 'images.txt').write_text('\n\n'.join(records) + '\n\n')
    (sparse / 'points3D.txt').write_text('# Shared initialization\n')
    with (sparse / 'cameras.bin').open('wb') as handle:
        handle.write(struct.pack('<Q', len(rows)))
        for camera_id, row in enumerate(rows, start=1):
            intrinsics = np.array(row['k'])
            handle.write(struct.pack('<iiQQ', camera_id, 4, 832, 480))
            handle.write(struct.pack('<8d', intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2], 0, 0, 0, 0))
    with (sparse / 'images.bin').open('wb') as handle:
        handle.write(struct.pack('<Q', len(rows)))
        for camera_id, row in enumerate(rows, start=1):
            pose = np.array(row['w2c'])
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            quaternion = np.r_[quaternion[3], quaternion[:3]]
            handle.write(struct.pack('<i7di', camera_id, *quaternion, *pose[:3, 3], camera_id))
            handle.write(row['name'].encode() + b'\0')
            handle.write(struct.pack('<Q', 0))
    shutil.copyfile(GROUP / 'initial_points.ply', root / 'point_cloud.ply')


def prepare():
    group_rows = read(GROUP / 'inputs.json')['frames']
    by_angle = {round(float(row['angle']), 2): row for row in group_rows if row['kind'] == 'generated'}
    real = [row for row in group_rows if row['kind'] == 'real']
    assert len(real) == 6 and all('w2c' in row and 'k' in row for row in real)
    rear_result = {round(float(frame['angle']), 2): frame for frame in read(REAR / 'result.json')['frames']}
    rear_trajectory = read(REAR / 'trajectory.json')
    rear_cameras = {round(float(angle), 2): (w2c, k) for angle, w2c, k in zip(rear_trajectory['target_angles_deg'], rear_trajectory['target_w2c'], rear_trajectory['target_intrinsics'])}
    for directory in (OUT / 'images', OUT / 'masks'):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in real:
        name = row['name']
        for folder in ('images', 'masks'):
            destination = OUT / folder / name
            source = GROUP / folder / name
            if not destination.exists():
                shutil.copyfile(source, destination)
            assert digest(destination) == digest(source)
        rows.append(dict(row, name=name))
    for offset, angle in enumerate(PRIOR_ANGLES):
        source_row = by_angle[angle]
        name = f'{6 + offset:02d}.png'
        for folder in ('images', 'masks'):
            destination = OUT / folder / name
            source = GROUP / folder / source_row['name']
            if not destination.exists():
                shutil.copyfile(source, destination)
            assert digest(destination) == digest(source)
        rows.append(dict(source_row, name=name, index=6 + offset))
    for offset, angle in enumerate(REAR_ANGLES):
        image_rel, mask_rel = REAR_FILES[angle]
        image = REAR / image_rel
        mask = REAR / mask_rel
        assert digest(image) == rear_result[angle]['sha256']
        with Image.open(image) as im, Image.open(mask) as mk:
            assert im.size == (832, 480) and mk.size == (832, 480)
        name = f'{12 + offset:02d}.png'
        if not (OUT / 'images' / name).exists():
            shutil.copyfile(image, OUT / 'images' / name)
        if not (OUT / 'masks' / name).exists():
            shutil.copyfile(mask, OUT / 'masks' / name)
        assert digest(OUT / 'images' / name) == digest(image)
        w2c, intrinsics = rear_cameras[angle]
        if angle in by_angle:
            assert np.allclose(w2c, by_angle[angle]['w2c'], atol=1e-8, rtol=0)
            assert np.allclose(intrinsics, by_angle[angle]['k'], atol=1e-8, rtol=0)
        rows.append(dict(name=name, kind='generated', angle=angle, index=12 + offset, source=str(image), sha256=digest(image), w2c=w2c, k=intrinsics))
    assert [row['name'] for row in rows] == [f'{i:02d}.png' for i in range(17)]
    assert [row['angle'] for row in rows if row['kind'] == 'generated'] == PRIOR_ANGLES + REAR_ANGLES
    if not (OUT / 'warmup_checkpoint.pt').exists():
        shutil.copyfile(GROUP / 'warmup_checkpoint.pt', OUT / 'warmup_checkpoint.pt')
    if not (OUT / 'initial_points.ply').exists():
        shutil.copyfile(GROUP / 'initial_points.ply', OUT / 'initial_points.ply')
    assert digest(OUT / 'warmup_checkpoint.pt') == digest(GROUP / 'warmup_checkpoint.pt')
    save(OUT / 'inputs.json', dict(frames=rows, width=832, height=480, prior_angles=PRIOR_ANGLES, rear_angles=REAR_ANGLES, generated_loss_weight=0.3, group_sampling='equal total probability', loss='same as groupa_tight_v1'))
    write_colmap(OUT / 'datasets/rear5', rows)
    print('PREPARED', len(rows), 'images', flush=True)


class RearTrainer(TightTrainer):
    def init_dataloaders(self, conf):
        Trainer3DGRUT.init_dataloaders(self, conf)
        stems = [int(Path(path).stem) for path in self.train_dataset.image_paths]
        assert stems == list(range(17)), stems
        weights = [1 / 6 if stem < 6 else 1 / 11 for stem in stems]
        assert abs(sum(weights[:6]) - 1) < 1e-8 and abs(sum(weights[6:]) - 1) < 1e-8
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(20261021))
        self.train_dataloader = MultiEpochsDataLoader(self.train_dataset, batch_size=1, num_workers=0, sampler=sampler, pin_memory=False)

    def init_scene_extents(self, dataset):
        rows = read(OUT / 'inputs.json')['frames']
        centers = np.linalg.inv(np.array([row['w2c'] for row in rows]))[:, :3, 3]
        self.scene_extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        bounds = read(GROUP / 'initialization.json')['bounds_world']
        self.scene_bbox = tuple(torch.tensor(value, device=self.device, dtype=torch.float32) for value in bounds)


def configuration():
    overrides = [f'path={OUT / "datasets/rear5"}', f'out_dir={OUT / "training"}', 'experiment_name=rear5',
                 'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={OUT / "initial_points.ply"}',
                 'n_iterations=5000', 'num_workers=0', 'enable_writer=false', 'use_wandb=false',
                 'test_last=false', 'compute_extra_metrics=false', 'export_ingp.enabled=false', 'export_ply.enabled=true',
                 f'export_ply.path={OUT / "rear5.ply"}', 'dataset.test_split_interval=0', 'val_frequency=999999',
                 'checkpoint.iterations=[1000,3000,5000]', 'model.progressive_training.max_n_features=1',
                 'scheduler.positions.max_steps=5000', 'strategy.densify.end_iteration=3000', 'strategy.prune.end_iteration=4500',
                 'strategy.reset_density.end_iteration=-1', '+car15_stage=rear5', f'resume={OUT / "warmup_checkpoint.pt"}']
    return compose_3dgrut_config('apps/colmap_3dgut', overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)


def train():
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    conf = configuration()
    (OUT / 'training').mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, OUT / 'training/rear5_config.yaml')
    start = time.time()
    trainer = RearTrainer(conf)
    assert trainer.global_step == 1000
    print('TRAIN_START', trainer.model.num_gaussians, flush=True)
    trainer.run_training()
    trainer.save_checkpoint()
    checkpoint = Path(trainer.tracking.output_dir) / f'ours_{trainer.global_step}' / f'ckpt_{trainer.global_step}.pt'
    save(OUT / 'training/rear5_result.json', dict(
        stage='rear5', step=trainer.global_step, gaussians=trainer.model.num_gaussians,
        elapsed_seconds=time.time() - start, checkpoint=str(checkpoint), ply=str(OUT / 'rear5.ply'),
        ply_sha256=digest(OUT / 'rear5.ply'), prior_angles=PRIOR_ANGLES, rear_angles=REAR_ANGLES,
        ratio_cap=RATIO_CAP, anisotropy_weight=ANISO_WEIGHT, outside_weight=OUTSIDE_WEIGHT,
        generated_loss_weight=0.3, warmup_sha256=digest(OUT / 'warmup_checkpoint.pt')))
    print('TRAIN_COMPLETED', trainer.global_step, trainer.model.num_gaussians, flush=True)


def load():
    result = read(OUT / 'training/rear5_result.json')
    conf = OmegaConf.load(OUT / 'training/rear5_config.yaml')
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(torch.load(result['checkpoint'], map_location='cuda', weights_only=False), setup_optimizer=False)
    model.eval()
    return model


def evaluate():
    torch.set_num_threads(4)
    root = OUT / 'evaluation'
    model = load()
    rows = []
    for frame in read(OUT / 'inputs.json')['frames']:
        rgb, alpha = keep.render(model, np.array(frame['w2c']), frame['k'])
        ground_truth = np.array(Image.open(OUT / 'images' / frame['name'])) / 255.
        mask = np.array(Image.open(OUT / 'masks' / frame['name'])) > 127
        rows.append(dict(name=frame['name'], kind=frame['kind'], angle=frame.get('angle'), **keep.metrics(rgb, alpha, ground_truth, mask)))
        if frame['kind'] == 'generated' and frame['angle'] in REAR_ANGLES:
            keep.write(root / 'supervised' / f"{frame['angle']:.2f}.png", rgb)
    validation = read(keep.OLD / 'validation/cameras.json')
    if isinstance(validation, dict):
        validation = validation.get('frames', validation.get('cameras', validation))
    validation_rows = []
    for frame in validation:
        name = Path(frame['name']).name
        rgb, alpha = keep.render(model, np.array(frame['w2c']), frame.get('k', frame.get('intrinsics')))
        ground_truth = np.array(Image.open(keep.OLD / 'validation/images' / name)) / 255.
        mask = np.array(Image.open(keep.OLD / 'validation/masks' / name)) > 127
        validation_rows.append(dict(name=name, **keep.metrics(rgb, alpha, ground_truth, mask)))
    orbit = read(BASE / 'vggtdiff_realvideo_orbit360_64_v2_batch8/trajectory.json')
    for index, (w2c, intrinsics) in enumerate(zip(orbit['target_w2c'], orbit['target_intrinsics'])):
        rgb, _ = keep.render(model, np.array(w2c), intrinsics)
        keep.write(root / 'orbit' / f'{index:03d}.png', rgb)
    del model
    torch.cuda.empty_cache()
    previous = BASE / 'vggtdiff_car15_keep6_v1/evaluation/orbit'
    for index in range(64):
        angle = index * 360 / 64
        keep.pair(previous / f'{index:03d}.png', root / 'orbit' / f'{index:03d}.png', root / 'versus_keep6' / f'{index:03d}.png', f'keep6 | {angle:.3f} deg', f'rear5 | {angle:.3f} deg')
    subprocess.run(['ffmpeg', '-y', '-framerate', '8', '-i', str(root / 'versus_keep6/%03d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(root / 'orbit_versus_keep6.mp4')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    selected = [16, 20, 22, 24, 27, 32]
    keep.e.contact([root / 'versus_keep6' / f'{index:03d}.png' for index in selected], [f'{index * 5.625:.2f} deg' for index in selected], root / 'right_side_contact.jpg', 2)
    ply = PlyData.read(OUT / 'rear5.ply')['vertex']
    scales = np.exp(np.stack([ply['scale_0'], ply['scale_1'], ply['scale_2']], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-ply['opacity'].astype(np.float64)))
    ratio = scales.max(1) / np.clip(scales.min(1), 1e-12, None)

    def mean(items, key):
        return float(np.mean([item[key] for item in items])) if items else None

    real = [row for row in rows if row['kind'] == 'real']
    generated = [row for row in rows if row['kind'] == 'generated']
    rear = [row for row in generated if row['angle'] in REAR_ANGLES]
    report = dict(gaussians=len(ply), elongated_opacity_mass_ratio16=float(opacity[ratio > 16].sum() / opacity.sum()),
                  real_training=dict(foreground_psnr=mean(real, 'foreground_psnr'), silhouette_iou=mean(real, 'silhouette_iou')),
                  generated_training=dict(foreground_psnr=mean(generated, 'foreground_psnr'), silhouette_iou=mean(generated, 'silhouette_iou')),
                  rear_training=dict(foreground_psnr=mean(rear, 'foreground_psnr'), silhouette_iou=mean(rear, 'silhouette_iou')),
                  independent_real=dict(foreground_psnr=mean(validation_rows, 'foreground_psnr'), silhouette_iou=mean(validation_rows, 'silhouette_iou')),
                  frames=rows)
    save(root / 'metrics.json', report)
    print('EVALUATION_COMPLETED', {key: value for key, value in report.items() if key != 'frames'}, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate', 'all'])
    args = parser.parse_args()
    if args.action in ('prepare', 'all'):
        prepare()
    if args.action in ('train', 'all'):
        train()
    if args.action in ('evaluate', 'all'):
        evaluate()
