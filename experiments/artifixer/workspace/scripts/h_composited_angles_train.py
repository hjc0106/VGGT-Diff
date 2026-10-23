"""Train on H_latest pred_composited frames at the rear5 supervision angles.

The 64 composited frames are one full orbit, 5.625 degrees apart, starting at
the chain's front camera. 120, 127.5 and 150 degrees fall between frames, so
those three use the nearest frame. Cameras stay in this orbit's own frame.
There is no real-photo group here, so every selected image has loss weight 1.
The silhouette and anisotropy terms match the tight rear5 run.
"""
import math
import shutil
import struct
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial.transform import Rotation

import vggtdiff_car15_train as video_train
from vggtdiff_car15_groupa_tight_train import ANISO_WEIGHT, OUTSIDE_WEIGHT, RATIO_CAP
from vggtdiff_white_car import BASE, digest, save
from car15_object_loss import object_losses
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.datasets.utils import MultiEpochsDataLoader
from threedgrut.model.losses import ssim
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch
from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR

OUT = BASE / 'h_pred_composited_angles_v1'
COMP = BASE / 'aggregated/H_latest/pred_composited'
TRANSFORMS = BASE / 'prepared/H/realworld_white_car_chain64_H_E4_front_views/transforms.json'
POINTS = BASE / 'source_64/realworld_white_car_cleanup_64_start19_front_views/initial_object/object_local.ply'
REQUESTED = [45.0, 56.25, 67.5, 78.75, 90.0, 101.25, 112.5, 120.0, 127.5, 135.0, 150.0]
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
SH_C0 = 0.28209479177387814


def read(path):
    import json
    return json.loads(Path(path).read_text())


def nearest_frame(angle):
    index = int(round(angle / 5.625)) % 64
    return index, index * 5.625


def opencv_w2c(opengl_c2w):
    c2w = np.asarray(opengl_c2w, dtype=np.float64) @ FLIP
    w2c = np.linalg.inv(c2w)
    if abs(np.linalg.det(w2c[:3, :3]) - 1) > 1e-5:
        raise ValueError('Camera rotation is not a proper rotation')
    return w2c


def car_mask(image):
    foreground = np.asarray(image).max(axis=2) >= 16
    labeled, count = ndimage.label(foreground)
    if count < 1:
        raise ValueError('No foreground in composited image')
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(labeled == sizes.argmax())


def write_points():
    source = PlyData.read(POINTS)['vertex']
    rgb = np.clip(0.5 + SH_C0 * np.stack([source['f_dc_0'], source['f_dc_1'], source['f_dc_2']], 1), 0, 1)
    rgb = np.uint8(rgb * 255)
    xyz = np.stack([source['x'], source['y'], source['z']], 1).astype(np.float32)
    records = np.empty(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    records['x'], records['y'], records['z'] = xyz.T
    records['red'], records['green'], records['blue'] = rgb.T
    destination = OUT / 'initial_points.ply'
    PlyData([PlyElement.describe(records, 'vertex')]).write(destination)
    return len(records)


def write_colmap(rows):
    root = OUT / 'datasets/hcomp'
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
        cameras.append(f'{camera_id} OPENCV 512 512 {intrinsics[0, 0]:.17g} {intrinsics[1, 1]:.17g} {intrinsics[0, 2]:.17g} {intrinsics[1, 2]:.17g} 0 0 0 0')
        records.append(' '.join(map(str, [camera_id, *quaternion, *pose[:3, 3], camera_id, row['name']])))
        shutil.copyfile(OUT / 'images' / row['name'], images / row['name'])
        shutil.copyfile(OUT / 'masks' / row['name'], images / f'{Path(row["name"]).stem}_mask.png')
    (sparse / 'cameras.txt').write_text('\n'.join(cameras) + '\n')
    (sparse / 'images.txt').write_text('\n\n'.join(records) + '\n\n')
    (sparse / 'points3D.txt').write_text('# Surface points are initialized from the source object ply\n')
    with (sparse / 'cameras.bin').open('wb') as handle:
        handle.write(struct.pack('<Q', len(rows)))
        for camera_id, row in enumerate(rows, start=1):
            intrinsics = np.array(row['k'])
            handle.write(struct.pack('<iiQQ', camera_id, 4, 512, 512))
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
    shutil.copyfile(OUT / 'initial_points.ply', root / 'point_cloud.ply')


def prepare():
    frames = read(TRANSFORMS)['frames']
    if len(frames) < 64:
        raise ValueError(f'Expected at least 64 orbit cameras, got {len(frames)}')
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'masks').mkdir(parents=True, exist_ok=True)
    chosen = []
    rows = []
    for slot, angle in enumerate(REQUESTED):
        index, actual = nearest_frame(angle)
        if any(item['frame'] == index for item in chosen):
            raise ValueError(f'Frame {index} was selected twice')
        frame = frames[index]
        image_path = COMP / f'{index:05d}.png'
        with Image.open(image_path) as im:
            rgb = im.convert('RGB')
            if rgb.size != (512, 512):
                raise ValueError(f'{image_path} is {rgb.size}')
            mask = car_mask(rgb)
        name = f'{slot:02d}.png'
        rgb.save(OUT / 'images' / name)
        Image.fromarray(mask.astype(np.uint8) * 255).save(OUT / 'masks' / name)
        intrinsics = np.array([[frame['fl_x'], 0, frame['cx']], [0, frame['fl_y'], frame['cy']], [0, 0, 1]], dtype=np.float64)
        ys, xs = np.where(mask)
        rows.append(dict(name=name, kind='composited', requested_angle=angle, angle=actual, frame=index,
                         source=str(image_path), sha256=digest(image_path), w2c=opencv_w2c(frame['transform_matrix']).tolist(),
                         k=intrinsics.tolist(), mask_width=int(xs.max() - xs.min()), mask_area=float(mask.mean())))
        chosen.append(rows[-1])
        print('SELECT', f'requested {angle:g}', f'frame {index:02d}', f'actual {actual:.3f}', f'width {rows[-1]["mask_width"]}', flush=True)
    count = write_points()
    save(OUT / 'inputs.json', dict(frames=rows, width=512, height=512, requested_angles=REQUESTED,
                                   note='Nearest frame on the 5.625 degree H orbit. All images are pred_composited, loss weight 1.',
                                   source_points=count))
    write_colmap(rows)
    sheet = Image.new('RGB', (256 * 4, 288 * 3), '#181818')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        image = np.array(Image.open(OUT / 'images' / row['name']))
        mask = np.array(Image.open(OUT / 'masks' / row['name'])) > 127
        image[mask] = (image[mask] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
        tile = Image.fromarray(image).resize((256, 256))
        x, y = (i % 4) * 256, (i // 4) * 288
        sheet.paste(tile, (x, y + 24))
        draw.text((x + 8, y + 6), f"{row['requested_angle']:g} -> {row['angle']:.2f}", fill='white')
    sheet.save(OUT / 'selection_contact.jpg', quality=95)
    print('PREPARED', len(rows), 'images', count, 'points', flush=True)


class HTrainer(video_train.ObjectTrainer):
    def setup_training(self, conf, model, train_dataset):
        Trainer3DGRUT.setup_training(self, conf, model, train_dataset)

    def init_dataloaders(self, conf):
        Trainer3DGRUT.init_dataloaders(self, conf)
        stems = [int(Path(path).stem) for path in self.train_dataset.image_paths]
        if stems != list(range(len(REQUESTED))):
            raise ValueError(f'Unexpected image order {stems}')
        weights = [1 / len(stems) for _ in stems]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(20261021))
        self.train_dataloader = MultiEpochsDataLoader(self.train_dataset, batch_size=1, num_workers=0, sampler=sampler, pin_memory=False)

    def init_scene_extents(self, dataset):
        rows = read(OUT / 'inputs.json')['frames']
        centers = np.linalg.inv(np.array([row['w2c'] for row in rows]))[:, :3, 3]
        self.scene_extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        points = np.stack([PlyData.read(OUT / 'initial_points.ply')['vertex'][axis] for axis in 'xyz'], 1)
        self.scene_bbox = tuple(torch.tensor(value, device=self.device, dtype=torch.float32) for value in (points.min(0), points.max(0)))

    def get_losses(self, batch, outputs):
        l1, struct, alpha = object_losses(outputs['pred_rgb'], batch.rgb_gt, outputs['pred_opacity'], batch.mask, ssim_fn=ssim, dilate=3)
        scales = self.model.get_scale()
        opacity = self.model.get_density().reshape(-1)
        ratio = scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(1e-8)
        aniso = (torch.relu(torch.log(ratio) - math.log(RATIO_CAP)) * opacity).sum() / opacity.sum().clamp_min(1e-6)
        total = 1.0 * (0.8 * l1 + 0.2 * struct + OUTSIDE_WEIGHT * alpha) + ANISO_WEIGHT * aniso
        zero = total.new_zeros(())
        return dict(total_loss=total, l1_loss=l1, l2_loss=zero, ssim_loss=struct, opacity_loss=alpha, scale_loss=aniso, lpips_loss=zero)


def configuration():
    overrides = [f'path={OUT / "datasets/hcomp"}', f'out_dir={OUT / "training"}', 'experiment_name=hcomp',
                 'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={OUT / "initial_points.ply"}',
                 'n_iterations=5000', 'num_workers=0', 'enable_writer=false', 'use_wandb=false',
                 'test_last=false', 'compute_extra_metrics=false', 'export_ingp.enabled=false', 'export_ply.enabled=true',
                 f'export_ply.path={OUT / "hcomp.ply"}', 'dataset.test_split_interval=0', 'val_frequency=999999',
                 'checkpoint.iterations=[1000,3000,5000]', 'model.progressive_training.max_n_features=1',
                 'scheduler.positions.max_steps=5000', 'strategy.densify.end_iteration=3000', 'strategy.prune.end_iteration=4500',
                 'strategy.reset_density.end_iteration=-1', '+car15_stage=hcomp']
    return compose_3dgrut_config('apps/colmap_3dgut', overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)


def train():
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    conf = configuration()
    (OUT / 'training').mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, OUT / 'training/hcomp_config.yaml')
    start = time.time()
    trainer = HTrainer(conf)
    if trainer.global_step != 0:
        raise RuntimeError(f'Expected a fresh run, got step {trainer.global_step}')
    print('TRAIN_START', trainer.model.num_gaussians, flush=True)
    trainer.run_training()
    trainer.save_checkpoint()
    checkpoint = Path(trainer.tracking.output_dir) / f'ours_{trainer.global_step}' / f'ckpt_{trainer.global_step}.pt'
    save(OUT / 'training/hcomp_result.json', dict(
        stage='hcomp', step=trainer.global_step, gaussians=trainer.model.num_gaussians,
        elapsed_seconds=time.time() - start, checkpoint=str(checkpoint), ply=str(OUT / 'hcomp.ply'),
        ply_sha256=digest(OUT / 'hcomp.ply'), requested_angles=REQUESTED,
        ratio_cap=RATIO_CAP, anisotropy_weight=ANISO_WEIGHT, outside_weight=OUTSIDE_WEIGHT,
        image_loss_weight=1.0))
    print('TRAIN_COMPLETED', trainer.global_step, trainer.model.num_gaussians, flush=True)


def load():
    result = read(OUT / 'training/hcomp_result.json')
    conf = OmegaConf.load(OUT / 'training/hcomp_config.yaml')
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(torch.load(result['checkpoint'], map_location='cuda', weights_only=False), setup_optimizer=False)
    model.eval()
    return model


@torch.inference_mode()
def render(model, w2c, intrinsics):
    intrinsics = np.array(intrinsics)
    blank = torch.zeros((1, 512, 512, 3), device='cuda')
    batch = Batch(blank, blank, torch.tensor(np.linalg.inv(w2c), dtype=torch.float32, device='cuda')[None], intrinsics_OpenCVPinholeCameraModelParameters=dict(focal_length=[intrinsics[0, 0], intrinsics[1, 1]], principal_point=[intrinsics[0, 2], intrinsics[1, 2]]))
    output = model(batch, train=False)
    return output['pred_rgb'][0].clamp(0, 1).cpu().numpy(), output['pred_opacity'][0, :, :, 0].cpu().numpy()


def metrics(rgb, alpha, ground_truth, mask):
    error = (rgb - ground_truth)[mask]
    mse = float(np.mean(error ** 2))
    solid = alpha > 0.5
    return dict(foreground_psnr=-10 * math.log10(max(mse, 1e-12)), silhouette_iou=float((solid & mask).sum() / max((solid | mask).sum(), 1)))


def evaluate():
    torch.set_num_threads(4)
    root = OUT / 'evaluation'
    model = load()
    frames = read(TRANSFORMS)['frames']
    trained = {row['frame']: row for row in read(OUT / 'inputs.json')['frames']}
    rows = []
    for index, frame in enumerate(frames[:64]):
        w2c = opencv_w2c(frame['transform_matrix'])
        intrinsics = np.array([[frame['fl_x'], 0, frame['cx']], [0, frame['fl_y'], frame['cy']], [0, 0, 1]])
        rgb, alpha = render(model, w2c, intrinsics)
        path = root / 'orbit' / f'{index:03d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(path)
        with Image.open(COMP / f'{index:05d}.png') as im:
            reference = np.array(im.convert('RGB')) / 255
            mask = car_mask(im.convert('RGB'))
        item = dict(frame=index, angle=index * 5.625, supervised=index in trained, **metrics(rgb, alpha, reference, mask))
        rows.append(item)
    del model
    torch.cuda.empty_cache()
    selected = [0, 8, 16, 24, 27, 32]
    sheet = Image.new('RGB', (512 * 2, 544 * len(selected)))
    draw = ImageDraw.Draw(sheet)
    for row_index, frame_index in enumerate(selected):
        y = row_index * 544
        sheet.paste(Image.open(COMP / f'{frame_index:05d}.png'), (0, y + 32))
        sheet.paste(Image.open(root / 'orbit' / f'{frame_index:03d}.png'), (512, y + 32))
        label = 'supervised' if frame_index in trained else 'held out'
        draw.text((8, y + 8), f'composited {frame_index * 5.625:.2f} deg', fill='white')
        draw.text((520, y + 8), f'reconstruction {label}', fill='white')
    sheet.save(root / 'angle_contact.jpg', quality=95)
    ply = PlyData.read(OUT / 'hcomp.ply')['vertex']
    scales = np.exp(np.stack([ply['scale_0'], ply['scale_1'], ply['scale_2']], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-ply['opacity'].astype(np.float64)))
    ratio = scales.max(1) / np.clip(scales.min(1), 1e-12, None)

    def mean(items, key):
        return float(np.mean([item[key] for item in items])) if items else None

    supervised = [row for row in rows if row['supervised']]
    held = [row for row in rows if not row['supervised']]
    report = dict(gaussians=len(ply), elongated_opacity_mass_ratio16=float(opacity[ratio > 16].sum() / opacity.sum()),
                  supervised=dict(foreground_psnr=mean(supervised, 'foreground_psnr'), silhouette_iou=mean(supervised, 'silhouette_iou')),
                  heldout=dict(foreground_psnr=mean(held, 'foreground_psnr'), silhouette_iou=mean(held, 'silhouette_iou')),
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
