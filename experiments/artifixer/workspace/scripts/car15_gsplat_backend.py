"""Standard 3DGS rasterization adapter for the existing 3DGRUT optimizer/model.

Avoids incompatible Slang JIT kernels; no vendor files are modified.
"""
import torch
from gsplat import rasterization


class GsplatBackend:
    def __init__(self,conf):self.conf=conf;self.timings={}
    def build_acc(self,model,rebuild=True):pass
    def render(self,model,batch,train=False,frame_id=0):
        params=batch.intrinsics_OpenCVPinholeCameraModelParameters
        if params is None:raise ValueError('The car15 experiment requires pinhole intrinsics')
        device=model.positions.device
        f=torch.as_tensor(params['focal_length'],device=device,dtype=torch.float32)
        c=torch.as_tensor(params['principal_point'],device=device,dtype=torch.float32)
        k=torch.eye(3,device=device);k[0,0]=f[0];k[1,1]=f[1];k[0,2]=c[0];k[1,2]=c[1]
        height,width=batch.rays_dir.shape[1:3]
        rgbd,alpha,info=rasterization(means=model.get_positions(),quats=model.get_rotation(),scales=model.get_scale(),
            opacities=model.get_density().flatten(),colors=model.get_features().reshape(model.num_gaussians,-1,3),
            viewmats=torch.linalg.inv(batch.T_to_world).contiguous(),Ks=k[None],width=width,height=height,
            sh_degree=int(model.n_active_features),packed=False,near_plane=.001,render_mode='RGB+ED')
        rgb=rgbd[:,:,:,:3]
        rgb,alpha=model.background(batch.T_to_world,batch.rays_dir,rgb,alpha,train)
        return dict(pred_rgb=rgb,pred_opacity=alpha,pred_dist=rgbd[:,:,:,3:4],mog_visibility=(info['radii']>0).any(0)[:,None])


def install():
    import threedgut_tracer
    threedgut_tracer.Tracer=GsplatBackend
