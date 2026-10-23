"""Object-only 3DGRUT experiment with explicit foreground and opacity losses."""
import sys,argparse,json,time,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from vggtdiff_car15_prepare import OUT,REPO,read
from vggtdiff_white_car import save,digest
sys.path[:0]=[str(REPO),str(REPO/'thirdparty/3DGRUT-ArtiFixer')]
from car15_gsplat_backend import install
install()
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.model.losses import ssim
from threedgrut.datasets.utils import MultiEpochsDataLoader
from data_processing.threedgrut_training import compose_3dgrut_config,DEFAULT_THREEDGRUT_CONFIG_DIR


from car15_object_loss import object_losses


class ObjectTrainer(Trainer3DGRUT):
    def init_dataloaders(self,conf):
        super().init_dataloaders(conf)
        weights=[1/6 if int(Path(p).stem)<6 else 1/9 for p in self.train_dataset.image_paths]
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(weights),replacement=True,generator=torch.Generator().manual_seed(20261021))
        self.train_dataloader=MultiEpochsDataLoader(self.train_dataset,batch_size=1,num_workers=0,sampler=sampler,pin_memory=False)

    def init_scene_extents(self,dataset):
        rows=read(OUT/'inputs.json')['frames'];centers=np.linalg.inv(np.array([r['w2c'] for r in rows]))[:,:3,3]
        self.scene_extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
        bounds=read(OUT/'initialization.json')['bounds_world'];self.scene_bbox=tuple(torch.tensor(x,device=self.device,dtype=torch.float32) for x in bounds)

    def init_densification_and_pruning_strategy(self,conf):
        super().init_densification_and_pruning_strategy(conf)
        original=self.strategy.densify_gaussians
        def limited(scene_extent):
            if self.model.num_gaussians<100000:
                original(scene_extent)
                if self.model.num_gaussians>100000:
                    with torch.no_grad():
                        keep=torch.zeros(self.model.num_gaussians,device=self.device,dtype=torch.bool)
                        keep[self.model.get_density().flatten().topk(100000).indices]=True
                        strategy=self.strategy
                        strategy._update_param_with_optimizer(lambda name,p:torch.nn.Parameter(p[keep],requires_grad=p.requires_grad),lambda key,v:v[keep])
                        strategy._filter_local_weights(keep);strategy.prune_densification_buffers(keep)
        self.strategy.densify_gaussians=limited

    def setup_training(self,conf,model,train_dataset):
        super().setup_training(conf,model,train_dataset)
        if not conf.resume:
            count=read(OUT/'initialization.json')['real_surface_points']
            with torch.no_grad():model.density[count:]=float(np.log(.01/.99))

    def get_losses(self,batch,outputs):
        l1,struct,alpha=object_losses(outputs['pred_rgb'],batch.rgb_gt,outputs['pred_opacity'],batch.mask,ssim_fn=ssim)
        synthetic=int(Path(batch.image_name).stem)>=6
        weight=.3 if synthetic else 1.
        scale=self.model.get_scale().mean()/self.scene_extent
        total=weight*(.8*l1+.2*struct+.05*alpha)+1e-5*scale
        zero=total.new_zeros(())
        return dict(total_loss=total,l1_loss=l1,l2_loss=zero,ssim_loss=struct,opacity_loss=alpha,scale_loss=scale,lpips_loss=zero)

    def get_metrics(self,batch,outputs,losses,profilers,split='training',iteration=None):
        return {'losses':{k:float(v.detach()) for k,v in losses.items()}}

    def log_training_iter(self,batch,outputs,metrics,iteration=None):
        if self.global_step%100==0:
            row=dict(step=self.global_step,gaussians=self.model.num_gaussians,**metrics['losses'])
            with open(Path(self.conf.out_dir)/f'{self.conf.car15_stage}_progress.jsonl','a') as f:f.write(json.dumps(row)+'\n')
            print('STEP',self.conf.car15_stage,json.dumps(row),flush=True)


def configuration(stage,steps=None):
    warm=stage=='warmup';data='real6' if stage in ['warmup','real6'] else 'mixed15'
    overrides=[f'path={OUT/"datasets"/data}',f'out_dir={OUT/"training"}',f'experiment_name={stage}',
        'initialization=fused_point_cloud',f'initialization.fused_point_cloud_path={OUT/"initial_points.ply"}',
        'n_iterations='+str(steps or (1000 if warm else 5000)), 'num_workers=0','enable_writer=false','use_wandb=false',
        'test_last=false','compute_extra_metrics=false','export_ingp.enabled=false','export_ply.enabled=true',
        f'export_ply.path={OUT/(stage+".ply")}', 'dataset.test_split_interval=0','val_frequency=999999',
        'checkpoint.iterations=[1000,3000,5000]','model.progressive_training.max_n_features=1',
        'scheduler.positions.max_steps=5000','strategy.densify.end_iteration=3000','strategy.prune.end_iteration=4500',
        'strategy.reset_density.end_iteration=-1',f'+car15_stage={stage}']
    if not warm:
        checkpoint=OUT/'warmup_checkpoint.pt';assert checkpoint.exists();overrides.append(f'resume={checkpoint}')
    return compose_3dgrut_config('apps/colmap_3dgut',overrides,DEFAULT_THREEDGRUT_CONFIG_DIR)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['warmup','real6','mixed15']);p.add_argument('--steps',type=int);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261021);np.random.seed(20261021)
    conf=configuration(args.stage,args.steps);(OUT/'training').mkdir(exist_ok=True)
    OmegaConf.save(conf,OUT/'training'/f'{args.stage}_config.yaml')
    start=time.time();trainer=ObjectTrainer(conf);print('TRAIN_START',args.stage,'initial_gaussians',trainer.model.num_gaussians,flush=True)
    trainer.run_training();trainer.save_checkpoint()
    checkpoint=Path(trainer.tracking.output_dir)/f'ours_{trainer.global_step}'/f'ckpt_{trainer.global_step}.pt'
    if args.stage=='warmup':__import__('shutil').copyfile(checkpoint,OUT/'warmup_checkpoint.pt')
    save(OUT/'training'/f'{args.stage}_result.json',dict(stage=args.stage,step=trainer.global_step,gaussians=trainer.model.num_gaussians,elapsed_seconds=time.time()-start,checkpoint=str(checkpoint),ply=str(OUT/(args.stage+'.ply')),ply_sha256=digest(OUT/(args.stage+'.ply')),warmup_sha256=digest(OUT/'warmup_checkpoint.pt') if (OUT/'warmup_checkpoint.pt').exists() else None))
    print('TRAIN_COMPLETED',args.stage,flush=True)


if __name__=='__main__':main()
