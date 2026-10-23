"""Train the group-A supervision branch from the shared real-image warmup."""
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

import vggtdiff_car15_train as train
from vggtdiff_car15_groupa_prepare import OUT
from vggtdiff_white_car import digest, save
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.datasets.utils import MultiEpochsDataLoader
from data_processing.threedgrut_training import compose_3dgrut_config, DEFAULT_THREEDGRUT_CONFIG_DIR


def read(path):
    import json
    return json.loads(path.read_text())


class GroupATrainer(train.ObjectTrainer):
    def init_dataloaders(self, conf):
        Trainer3DGRUT.init_dataloaders(self, conf)
        weights = [1 / 6 if int(Path(path).stem) < 6 else 1 / 8 for path in self.train_dataset.image_paths]
        assert abs(sum(weights[:6]) - 1) < 1e-8 and abs(sum(weights[6:]) - 1) < 1e-8
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(20261021))
        self.train_dataloader = MultiEpochsDataLoader(self.train_dataset, batch_size=1, num_workers=0, sampler=sampler, pin_memory=False)

    def init_scene_extents(self, dataset):
        rows = read(OUT / 'inputs.json')['frames']
        centers = np.linalg.inv(np.array([row['w2c'] for row in rows]))[:, :3, 3]
        self.scene_extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        bounds = read(OUT / 'initialization.json')['bounds_world']
        self.scene_bbox = tuple(torch.tensor(value, device=self.device, dtype=torch.float32) for value in bounds)


def configuration():
    overrides = [f'path={OUT / "datasets/groupa"}', f'out_dir={OUT / "training"}', 'experiment_name=groupa',
                 'initialization=fused_point_cloud', f'initialization.fused_point_cloud_path={OUT / "initial_points.ply"}',
                 'n_iterations=5000', 'num_workers=0', 'enable_writer=false', 'use_wandb=false',
                 'test_last=false', 'compute_extra_metrics=false', 'export_ingp.enabled=false', 'export_ply.enabled=true',
                 f'export_ply.path={OUT / "groupa.ply"}', 'dataset.test_split_interval=0', 'val_frequency=999999',
                 'checkpoint.iterations=[1000,3000,5000]', 'model.progressive_training.max_n_features=1',
                 'scheduler.positions.max_steps=5000', 'strategy.densify.end_iteration=3000', 'strategy.prune.end_iteration=4500',
                 'strategy.reset_density.end_iteration=-1', '+car15_stage=groupa', f'resume={OUT / "warmup_checkpoint.pt"}']
    return compose_3dgrut_config('apps/colmap_3dgut', overrides, DEFAULT_THREEDGRUT_CONFIG_DIR)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    conf = configuration()
    (OUT / 'training').mkdir(exist_ok=True)
    OmegaConf.save(conf, OUT / 'training/groupa_config.yaml')
    start = time.time()
    trainer = GroupATrainer(conf)
    assert trainer.global_step == 1000
    print('TRAIN_START', 'groupa', 'initial_gaussians', trainer.model.num_gaussians, 'step', trainer.global_step, flush=True)
    trainer.run_training()
    trainer.save_checkpoint()
    checkpoint = Path(trainer.tracking.output_dir) / f'ours_{trainer.global_step}' / f'ckpt_{trainer.global_step}.pt'
    save(OUT / 'training/groupa_result.json', dict(stage='groupa', step=trainer.global_step, gaussians=trainer.model.num_gaussians, elapsed_seconds=time.time() - start, checkpoint=str(checkpoint), ply=str(OUT / 'groupa.ply'), ply_sha256=digest(OUT / 'groupa.ply'), warmup_sha256=digest(OUT / 'warmup_checkpoint.pt')))
    print('TRAIN_COMPLETED', trainer.global_step, trainer.model.num_gaussians, flush=True)


if __name__ == '__main__':
    main()
