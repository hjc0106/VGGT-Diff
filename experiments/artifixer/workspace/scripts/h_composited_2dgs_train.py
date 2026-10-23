"""Retrain the eleven H composited views with 2DGS instead of 3DGS.

Same images, cameras, initialization, step count, and silhouette loss as
h_pred_composited_angles_v1. The rasterizer is 2DGS, with its normal
consistency term at the object-scene weight 0.05.
"""
import math
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from plyfile import PlyData

import car15_gsplat_2dgs_backend as twod
import h_composited_angles_train as base
from vggtdiff_car15_groupa_tight_train import ANISO_WEIGHT, OUTSIDE_WEIGHT, RATIO_CAP
from vggtdiff_white_car import digest, save
from car15_object_loss import object_losses
from threedgrut.model.losses import ssim
from threedgrut.model.model import MixtureOfGaussians
from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR

twod.install()
OUT = base.OUT.with_name('h_pred_composited_2dgs_v1')
NORMAL_WEIGHT = 0.05


class TwoDTrainer(base.HTrainer):
    def get_losses(self, batch, outputs):
        l1, struct, alpha = object_losses(outputs['pred_rgb'], batch.rgb_gt, outputs['pred_opacity'], batch.mask, ssim_fn=ssim, dilate=3)
        scales = self.model.get_scale()
        opacity = self.model.get_density().reshape(-1)
        ratio = scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(1e-8)
        aniso = (torch.relu(torch.log(ratio) - math.log(RATIO_CAP)) * opacity).sum() / opacity.sum().clamp_min(1e-6)
        normals = outputs['pred_normals']
        surface = outputs['surf_normals']
        if surface.ndim == 3:
            surface = surface.unsqueeze(0)
        weight = outputs['pred_opacity'].detach().squeeze(-1).clamp(0, 1)
        align = (normals * surface).sum(-1).clamp(-1, 1)
        normal = ((1 - align) * weight).sum() / weight.sum().clamp_min(1e-6)
        total = 0.8 * l1 + 0.2 * struct + OUTSIDE_WEIGHT * alpha + ANISO_WEIGHT * aniso + NORMAL_WEIGHT * normal
        zero = total.new_zeros(())
        return dict(total_loss=total, l1_loss=l1, l2_loss=zero, ssim_loss=struct, opacity_loss=alpha, scale_loss=aniso, lpips_loss=normal)


def configuration():
    overrides = [f'path={base.OUT / "datasets/hcomp"}', f'out_dir={OUT / "training"}', 'experiment_name=h2dgs',
                 'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={base.OUT / "initial_points.ply"}',
                 'n_iterations=5000', 'num_workers=0', 'enable_writer=false', 'use_wandb=false',
                 'test_last=false', 'compute_extra_metrics=false', 'export_ingp.enabled=false', 'export_ply.enabled=true',
                 f'export_ply.path={OUT / "h2dgs.ply"}', 'dataset.test_split_interval=0', 'val_frequency=999999',
                 'checkpoint.iterations=[1000,3000,5000]', 'model.progressive_training.max_n_features=1',
                 'scheduler.positions.max_steps=5000', 'strategy.densify.end_iteration=3000', 'strategy.prune.end_iteration=4500',
                 'strategy.reset_density.end_iteration=-1', '+car15_stage=h2dgs']
    return compose_3dgrut_config('apps/colmap_3dgut', overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)


def train():
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    conf = configuration()
    (OUT / 'training').mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, OUT / 'training/h2dgs_config.yaml')
    start = time.time()
    trainer = TwoDTrainer(conf)
    if trainer.global_step != 0:
        raise RuntimeError(f'Expected a fresh run, got step {trainer.global_step}')
    print('TRAIN_START', trainer.model.num_gaussians, flush=True)
    trainer.run_training()
    trainer.save_checkpoint()
    checkpoint = Path(trainer.tracking.output_dir) / f'ours_{trainer.global_step}' / f'ckpt_{trainer.global_step}.pt'
    save(OUT / 'training/h2dgs_result.json', dict(
        stage='h2dgs', step=trainer.global_step, gaussians=trainer.model.num_gaussians,
        elapsed_seconds=time.time() - start, checkpoint=str(checkpoint), ply=str(OUT / 'h2dgs.ply'),
        ply_sha256=digest(OUT / 'h2dgs.ply'), rasterizer='gsplat 2DGS', normal_weight=NORMAL_WEIGHT,
        ratio_cap=RATIO_CAP, anisotropy_weight=ANISO_WEIGHT, outside_weight=OUTSIDE_WEIGHT))
    print('TRAIN_COMPLETED', trainer.global_step, trainer.model.num_gaussians, flush=True)


def load():
    result = base.read(OUT / 'training/h2dgs_result.json')
    conf = OmegaConf.load(OUT / 'training/h2dgs_config.yaml')
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(torch.load(result['checkpoint'], map_location='cuda', weights_only=False), setup_optimizer=False)
    model.eval()
    return model


def evaluate():
    torch.set_num_threads(4)
    root = OUT / 'evaluation'
    model = load()
    frames = base.read(base.TRANSFORMS)['frames']
    trained = {row['frame']: row for row in base.read(base.OUT / 'inputs.json')['frames']}
    rows = []
    for index, frame in enumerate(frames[:64]):
        w2c = base.opencv_w2c(frame['transform_matrix'])
        intrinsics = np.array([[frame['fl_x'], 0, frame['cx']], [0, frame['fl_y'], frame['cy']], [0, 0, 1]])
        rgb, alpha = base.render(model, w2c, intrinsics)
        path = root / 'orbit' / f'{index:03d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(path)
        with Image.open(base.COMP / f'{index:05d}.png') as im:
            reference = np.array(im.convert('RGB')) / 255
            mask = base.car_mask(im.convert('RGB'))
        rows.append(dict(frame=index, angle=index * 5.625, supervised=index in trained, **base.metrics(rgb, alpha, reference, mask)))
    del model
    torch.cuda.empty_cache()
    selected = [0, 8, 16, 24, 27, 32]
    previous = base.OUT / 'evaluation/orbit'
    sheet = Image.new('RGB', (512 * 2, 544 * len(selected)))
    draw = ImageDraw.Draw(sheet)
    panels = root / 'versus_3dgs'
    panels.mkdir(parents=True, exist_ok=True)
    for index in range(64):
        panel = Image.new('RGB', (1024, 544))
        panel.paste(Image.open(previous / f'{index:03d}.png'), (0, 32))
        panel.paste(Image.open(root / 'orbit' / f'{index:03d}.png'), (512, 32))
        label = ImageDraw.Draw(panel)
        label.text((8, 8), f'3DGS {index * 5.625:.2f} deg', fill='white')
        label.text((520, 8), f'2DGS {index * 5.625:.2f} deg', fill='white')
        panel.save(panels / f'{index:03d}.png')
    for row_index, frame_index in enumerate(selected):
        y = row_index * 544
        sheet.paste(Image.open(panels / f'{frame_index:03d}.png').crop((0, 32, 512, 544)), (0, y + 32))
        sheet.paste(Image.open(panels / f'{frame_index:03d}.png').crop((512, 32, 1024, 544)), (512, y + 32))
        draw.text((8, y + 8), f'3DGS {frame_index * 5.625:.2f}', fill='white')
        draw.text((520, y + 8), f'2DGS {frame_index * 5.625:.2f}', fill='white')
    sheet.save(root / 'angle_contact.jpg', quality=95)
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-framerate', '8', '-i', str(panels / '%03d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(root / 'orbit_3dgs_versus_2dgs.mp4')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ply = PlyData.read(OUT / 'h2dgs.ply')['vertex']
    scales = np.exp(np.stack([ply['scale_0'], ply['scale_1'], ply['scale_2']], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-ply['opacity'].astype(np.float64)))
    ratio = scales.max(1) / np.clip(scales.min(1), 1e-12, None)

    def mean(items, key):
        return float(np.mean([item[key] for item in items])) if items else None

    supervised = [row for row in rows if row['supervised']]
    held = [row for row in rows if not row['supervised']]
    report = dict(gaussians=len(ply), elongated_opacity_mass_ratio16=float(opacity[ratio > 16].sum() / opacity.sum()),
                  supervised=dict(foreground_psnr=mean(supervised, 'foreground_psnr'), silhouette_iou=mean(supervised, 'silhouette_iou')),
                  heldout=dict(foreground_psnr=mean(held, 'foreground_psnr'), silhouette_iou=mean(held, 'silhouette_iou')))
    save(root / 'metrics.json', report)
    print('EVALUATION_COMPLETED', report, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'evaluate', 'all'])
    args = parser.parse_args()
    if args.action in ('train', 'all'):
        train()
    if args.action in ('evaluate', 'all'):
        evaluate()
