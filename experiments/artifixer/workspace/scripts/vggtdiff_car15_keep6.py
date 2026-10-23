"""Train with the six consistent generated views and the existing tight loss.

Drops the group-A views at 112.5 and 135 degrees. Those cameras are rendered
after training for inspection and are not part of the loss. Real images, the
warmup checkpoint, initialization, and loss weights stay unchanged.
"""
import math
import shutil
import struct
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial.transform import Rotation

import vggtdiff_realvideo_unseen as e
from vggtdiff_car15_groupa_prepare import OLD, OUT as GROUP
from vggtdiff_car15_groupa_tight_train import ANISO_WEIGHT, OUTSIDE_WEIGHT, RATIO_CAP, TightTrainer
from vggtdiff_white_car import BASE, digest, save
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.datasets.utils import MultiEpochsDataLoader
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch
from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR

OUT = BASE / 'vggtdiff_car15_keep6_v1'
KEEP_ANGLES = [45.0, 56.25, 67.5, 78.75, 90.0, 101.25]
HOLDOUT_ANGLES = [112.5, 135.0]


def read(path):
    import json
    return json.loads(path.read_text())


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
    source_rows = read(GROUP / 'inputs.json')['frames']
    kept = [row for row in source_rows if row['kind'] == 'real' or row['angle'] in KEEP_ANGLES]
    held = [row for row in source_rows if row.get('angle') in HOLDOUT_ANGLES]
    assert [row.get('angle') for row in kept if row['kind'] == 'generated'] == KEEP_ANGLES
    assert [row['angle'] for row in held] == HOLDOUT_ANGLES
    for directory in [OUT / 'images', OUT / 'masks']:
        directory.mkdir(parents=True, exist_ok=True)
    for row in kept:
        for folder in ['images', 'masks']:
            destination = OUT / folder / row['name']
            source = GROUP / folder / row['name']
            if not destination.exists():
                shutil.copyfile(source, destination)
            assert digest(destination) == digest(source)
    if not (OUT / 'warmup_checkpoint.pt').exists():
        shutil.copyfile(GROUP / 'warmup_checkpoint.pt', OUT / 'warmup_checkpoint.pt')
    if not (OUT / 'initial_points.ply').exists():
        shutil.copyfile(GROUP / 'initial_points.ply', OUT / 'initial_points.ply')
    assert digest(OUT / 'warmup_checkpoint.pt') == digest(GROUP / 'warmup_checkpoint.pt')
    save(OUT / 'inputs.json', dict(frames=kept, width=832, height=480, kept_angles=KEEP_ANGLES, heldout_angles=HOLDOUT_ANGLES, loss='same as groupa_tight_v1'))
    save(OUT / 'heldout.json', dict(frames=held))
    write_colmap(OUT / 'datasets/keep6', kept)
    print('PREPARED', len(kept), 'train', len(held), 'heldout', flush=True)


class Keep6Trainer(TightTrainer):
    def init_dataloaders(self, conf):
        Trainer3DGRUT.init_dataloaders(self, conf)
        stems = [int(Path(path).stem) for path in self.train_dataset.image_paths]
        assert sorted(stems) == list(range(12)), stems
        weights = [1 / 6 for _ in stems]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(20261021))
        self.train_dataloader = MultiEpochsDataLoader(self.train_dataset, batch_size=1, num_workers=0, sampler=sampler, pin_memory=False)

    def init_scene_extents(self, dataset):
        rows = read(OUT / 'inputs.json')['frames']
        centers = np.linalg.inv(np.array([row['w2c'] for row in rows]))[:, :3, 3]
        self.scene_extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        bounds = read(GROUP / 'initialization.json')['bounds_world']
        self.scene_bbox = tuple(torch.tensor(value, device=self.device, dtype=torch.float32) for value in bounds)


def configuration():
    overrides = [f'path={OUT / "datasets/keep6"}', f'out_dir={OUT / "training"}', 'experiment_name=keep6',
                 'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={OUT / "initial_points.ply"}',
                 'n_iterations=5000', 'num_workers=0', 'enable_writer=false', 'use_wandb=false',
                 'test_last=false', 'compute_extra_metrics=false', 'export_ingp.enabled=false', 'export_ply.enabled=true',
                 f'export_ply.path={OUT / "keep6.ply"}', 'dataset.test_split_interval=0', 'val_frequency=999999',
                 'checkpoint.iterations=[1000,3000,5000]', 'model.progressive_training.max_n_features=1',
                 'scheduler.positions.max_steps=5000', 'strategy.densify.end_iteration=3000', 'strategy.prune.end_iteration=4500',
                 'strategy.reset_density.end_iteration=-1', '+car15_stage=keep6', f'resume={OUT / "warmup_checkpoint.pt"}']
    return compose_3dgrut_config('apps/colmap_3dgut', overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)


def train():
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    conf = configuration()
    (OUT / 'training').mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, OUT / 'training/keep6_config.yaml')
    start = time.time()
    trainer = Keep6Trainer(conf)
    assert trainer.global_step == 1000
    print('TRAIN_START', trainer.model.num_gaussians, flush=True)
    trainer.run_training()
    trainer.save_checkpoint()
    checkpoint = Path(trainer.tracking.output_dir) / f'ours_{trainer.global_step}' / f'ckpt_{trainer.global_step}.pt'
    save(OUT / 'training/keep6_result.json', dict(stage='keep6', step=trainer.global_step, gaussians=trainer.model.num_gaussians, elapsed_seconds=time.time() - start, checkpoint=str(checkpoint), ply=str(OUT / 'keep6.ply'), ply_sha256=digest(OUT / 'keep6.ply'), kept_angles=KEEP_ANGLES, heldout_angles=HOLDOUT_ANGLES, ratio_cap=RATIO_CAP, anisotropy_weight=ANISO_WEIGHT, outside_weight=OUTSIDE_WEIGHT, generated_loss_weight=0.3, warmup_sha256=digest(OUT / 'warmup_checkpoint.pt')))
    print('TRAIN_COMPLETED', trainer.global_step, trainer.model.num_gaussians, flush=True)


def load():
    result = read(OUT / 'training/keep6_result.json')
    conf = OmegaConf.load(OUT / 'training/keep6_config.yaml')
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(torch.load(result['checkpoint'], map_location='cuda', weights_only=False), setup_optimizer=False)
    model.eval()
    return model


@torch.inference_mode()
def render(model, w2c, intrinsics):
    intrinsics = np.array(intrinsics)
    blank = torch.zeros((1, 480, 832, 3), device='cuda')
    batch = Batch(blank, blank, torch.tensor(np.linalg.inv(w2c), dtype=torch.float32, device='cuda')[None], intrinsics_OpenCVPinholeCameraModelParameters=dict(focal_length=[intrinsics[0, 0], intrinsics[1, 1]], principal_point=[intrinsics[0, 2], intrinsics[1, 2]]))
    output = model(batch, train=False)
    return output['pred_rgb'][0].clamp(0, 1).cpu().numpy(), output['pred_opacity'][0, :, :, 0].cpu().numpy()


def write(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(path)


def metrics(rgb, alpha, ground_truth, mask):
    error = (rgb - ground_truth)[mask]
    mse = float(np.mean(error ** 2))
    solid = alpha > .5
    return dict(foreground_psnr=-10 * math.log10(max(mse, 1e-12)), silhouette_iou=float((solid & mask).sum() / max((solid | mask).sum(), 1)))


def pair(left, right, path, left_label, right_label):
    image = Image.new('RGB', (1664, 512))
    image.paste(Image.open(left), (0, 32))
    image.paste(Image.open(right), (832, 32))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), left_label, fill='white')
    draw.text((842, 10), right_label, fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def evaluate():
    torch.set_num_threads(4)
    root = OUT / 'evaluation'
    model = load()
    rows = []
    for frame in read(OUT / 'inputs.json')['frames']:
        rgb, alpha = render(model, np.array(frame['w2c']), frame['k'])
        ground_truth = np.array(Image.open(OUT / 'images' / frame['name'])) / 255.
        mask = np.array(Image.open(OUT / 'masks' / frame['name'])) > 127
        rows.append(dict(name=frame['name'], kind=frame['kind'], angle=frame.get('angle'), **metrics(rgb, alpha, ground_truth, mask)))
    validation = read(OLD / 'validation/cameras.json')
    if isinstance(validation, dict):
        validation = validation.get('frames', validation.get('cameras', validation))
    validation_rows = []
    for frame in validation:
        name = Path(frame['name']).name
        rgb, alpha = render(model, np.array(frame['w2c']), frame.get('k', frame.get('intrinsics')))
        ground_truth = np.array(Image.open(OLD / 'validation/images' / name)) / 255.
        mask = np.array(Image.open(OLD / 'validation/masks' / name)) > 127
        validation_rows.append(dict(name=name, **metrics(rgb, alpha, ground_truth, mask)))
    held_rows = []
    for frame in read(OUT / 'heldout.json')['frames']:
        rgb, alpha = render(model, np.array(frame['w2c']), frame['k'])
        write(root / 'heldout' / f"{frame['angle']:.2f}.png", rgb)
        ground_truth = np.array(Image.open(GROUP / 'images' / frame['name'])) / 255.
        mask = np.array(Image.open(GROUP / 'masks' / frame['name'])) > 127
        held_rows.append(dict(angle=frame['angle'], **metrics(rgb, alpha, ground_truth, mask)))
    orbit = read(BASE / 'vggtdiff_realvideo_orbit360_64_v2_batch8/trajectory.json')
    for index, (w2c, intrinsics) in enumerate(zip(orbit['target_w2c'], orbit['target_intrinsics'])):
        rgb, alpha = render(model, np.array(w2c), intrinsics)
        write(root / 'orbit' / f'{index:03d}.png', rgb)
    del model
    torch.cuda.empty_cache()
    previous = BASE / 'vggtdiff_car15_groupa_tight_v1/evaluation/orbit'
    for index in range(64):
        angle = index * 360 / 64
        pair(previous / f'{index:03d}.png', root / 'orbit' / f'{index:03d}.png', root / 'versus_tight' / f'{index:03d}.png', f'eight generated | {angle:.3f} deg', f'six generated | {angle:.3f} deg')
    subprocess.run(['ffmpeg', '-y', '-framerate', '8', '-i', str(root / 'versus_tight/%03d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(root / 'orbit_versus_tight.mp4')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    selected = [8, 16, 18, 20, 24, 32]
    e.contact([root / 'versus_tight' / f'{index:03d}.png' for index in selected], [f'{index * 5.625:.2f} deg' for index in selected], root / 'right_side_contact.jpg', 2)
    ply = PlyData.read(OUT / 'keep6.ply')['vertex']
    scales = np.exp(np.stack([ply['scale_0'], ply['scale_1'], ply['scale_2']], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-ply['opacity'].astype(np.float64)))
    ratio = scales.max(1) / np.clip(scales.min(1), 1e-12, None)

    def mean(items, key):
        return float(np.mean([item[key] for item in items])) if items else None

    real = [row for row in rows if row['kind'] == 'real']
    generated = [row for row in rows if row['kind'] == 'generated']
    report = dict(gaussians=len(ply), elongated_opacity_mass_ratio16=float(opacity[ratio > 16].sum() / opacity.sum()),
                  real_training=dict(foreground_psnr=mean(real, 'foreground_psnr'), silhouette_iou=mean(real, 'silhouette_iou')),
                  generated_training=dict(foreground_psnr=mean(generated, 'foreground_psnr'), silhouette_iou=mean(generated, 'silhouette_iou')),
                  independent_real=dict(foreground_psnr=mean(validation_rows, 'foreground_psnr'), silhouette_iou=mean(validation_rows, 'silhouette_iou')),
                  heldout=held_rows)
    save(root / 'metrics.json', report)
    print('EVALUATION_COMPLETED', report, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[args.action]()
