"""Retrain the group-A set with a cap on elongated Gaussians and a tighter silhouette."""
import math
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from vggtdiff_car15_groupa_prepare import OUT as GROUP
from vggtdiff_car15_groupa_train import GroupATrainer
from vggtdiff_white_car import BASE, digest, save
from car15_object_loss import object_losses
from threedgrut.model.losses import ssim
from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR

OUT = BASE / 'vggtdiff_car15_groupa_tight_v1'
RATIO_CAP = 16
ANISO_WEIGHT = 0.03
OUTSIDE_WEIGHT = 0.2


class TightTrainer(GroupATrainer):
    def get_losses(self, batch, outputs):
        l1, struct, alpha = object_losses(outputs['pred_rgb'], batch.rgb_gt, outputs['pred_opacity'], batch.mask, ssim_fn=ssim, dilate=3)
        synthetic = int(Path(batch.image_name).stem) >= 6
        weight = .3 if synthetic else 1.
        scales = self.model.get_scale()
        opacity = self.model.get_density().reshape(-1)
        ratio = scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(1e-8)
        aniso = (torch.relu(torch.log(ratio) - math.log(RATIO_CAP)) * opacity).sum() / opacity.sum().clamp_min(1e-6)
        total = weight * (.8 * l1 + .2 * struct + OUTSIDE_WEIGHT * alpha) + ANISO_WEIGHT * aniso
        zero = total.new_zeros(())
        return dict(total_loss=total, l1_loss=l1, l2_loss=zero, ssim_loss=struct, opacity_loss=alpha, scale_loss=aniso, lpips_loss=zero)


def configuration():
    overrides = [f'path={GROUP / "datasets/groupa"}', f'out_dir={OUT / "training"}', 'experiment_name=tight',
                 'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={GROUP / "initial_points.ply"}',
                 'n_iterations=5000', 'num_workers=0', 'enable_writer=false', 'use_wandb=false',
                 'test_last=false', 'compute_extra_metrics=false', 'export_ingp.enabled=false', 'export_ply.enabled=true',
                 f'export_ply.path={OUT / "tight.ply"}', 'dataset.test_split_interval=0', 'val_frequency=999999',
                 'checkpoint.iterations=[1000,3000,5000]', 'model.progressive_training.max_n_features=1',
                 'scheduler.positions.max_steps=5000', 'strategy.densify.end_iteration=3000', 'strategy.prune.end_iteration=4500',
                 'strategy.reset_density.end_iteration=-1', '+car15_stage=tight', f'resume={GROUP / "warmup_checkpoint.pt"}']
    return compose_3dgrut_config('apps/colmap_3dgut', overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    conf = configuration()
    (OUT / 'training').mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, OUT / 'training/tight_config.yaml')
    start = time.time()
    trainer = TightTrainer(conf)
    assert trainer.global_step == 1000
    print('TRAIN_START', 'tight', 'initial_gaussians', trainer.model.num_gaussians, flush=True)
    trainer.run_training()
    trainer.save_checkpoint()
    checkpoint = Path(trainer.tracking.output_dir) / f'ours_{trainer.global_step}' / f'ckpt_{trainer.global_step}.pt'
    save(OUT / 'training/tight_result.json', dict(stage='tight', step=trainer.global_step, gaussians=trainer.model.num_gaussians, elapsed_seconds=time.time() - start, checkpoint=str(checkpoint), ply=str(OUT / 'tight.ply'), ply_sha256=digest(OUT / 'tight.ply'), ratio_cap=RATIO_CAP, anisotropy_weight=ANISO_WEIGHT, outside_weight=OUTSIDE_WEIGHT, mask_dilation=3, generated_loss_weight=0.3, warmup_sha256=digest(GROUP / 'warmup_checkpoint.pt')))
    print('TRAIN_COMPLETED', trainer.global_step, trainer.model.num_gaussians, flush=True)


if __name__ == '__main__':
    main()
