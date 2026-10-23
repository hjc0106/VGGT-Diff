"""CPU projection + tiled compositor; validate against saved CUDA renders before use."""
from pathlib import Path
import ctypes
import subprocess
import numpy as np
import torch
from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci,_fully_fused_projection,_spherical_harmonics


class CPUReference:
    def __init__(self,checkpoint,out):
        src=Path(__file__).with_name('ghost_cpu_rasterizer.cpp')
        lib=out/'cpu_rasterizer.so'
        subprocess.run(['g++','-O3','-shared','-fPIC','-fopenmp',str(src),'-o',str(lib)],check=True)
        self.lib=ctypes.CDLL(str(lib));self.lib.composite.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*9
        self.lib.composite.restype=None
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.device='cpu'
        self.geometry=dict(means=state['positions'].detach(),quats=torch.nn.functional.normalize(state['rotation'].detach(),dim=-1),
            scales=state['scale'].detach().exp(),opacities=state['density'].detach().sigmoid().flatten())
        self.features=torch.cat([state['features_albedo'],state['features_specular']],dim=1).reshape(-1,4,3).detach()
        self.degree=int(state['n_active_features']);self.n=len(self.geometry['means'])
        self.covars,_=_quat_scale_to_covar_preci(self.geometry['quats'],self.geometry['scales'],compute_preci=False)
        self.cache={}

    def project(self,row):
        key=np.asarray(row['w2c']).tobytes()+np.asarray(row['k']).tobytes()
        if key not in self.cache:
            v=torch.tensor(row['w2c'],dtype=torch.float32)[None];k=torch.tensor(row['k'],dtype=torch.float32)[None]
            radii,xy,z,conic,_=_fully_fused_projection(self.geometry['means'],self.covars,v,k,832,480,near_plane=.001)
            center=torch.linalg.inv(v)[0,:3,3]
            dirs=self.geometry['means']-center
            colors=(_spherical_harmonics(self.degree,dirs,self.features)+.5).clamp_min(0)
            self.cache[key]=[np.ascontiguousarray(x[0].numpy()) for x in [xy,conic,radii,z]]+[np.ascontiguousarray(colors.numpy())]
        return self.cache[key]

    def compose(self,row,colors=None,remove=None,pixel_weights=None):
        xy,conic,radii,z,original=self.project(row)
        color=original if colors is None else np.ascontiguousarray(colors.detach().numpy(),dtype=np.float32)
        opacity=self.geometry['opacities'].numpy().copy()
        if remove is not None:opacity[np.asarray(remove)]=0
        weights=None if pixel_weights is None else np.ascontiguousarray(pixel_weights,dtype=np.float32)
        image=np.zeros((480,832,4),dtype=np.float32)
        mass=None if weights is None else np.zeros((self.n,3),dtype=np.float64)
        arrays=[xy,conic,radii,z,opacity,color,weights,image,mass]
        self.lib.composite(self.n,832,480,*[None if a is None else a.ctypes.data for a in arrays])
        return image,mass

    def rgb(self,row,remove=None):
        im,_=self.compose(row,remove=remove)
        return np.clip(im[:,:,:3],0,1),im[:,:,3]

    def attribution(self,row,pixel_weights):
        image,mass=self.compose(row,pixel_weights=pixel_weights)
        expected=(image[:,:,3,None]*pixel_weights).sum((0,1),dtype=np.float64)
        err=np.abs(mass.sum(0)-expected)
        assert (err<.01+expected*1e-4).all(),(err,expected)
        return mass.astype('float32'),image[:,:,3],float(err.max())

    def tagged(self,row,labels):
        im,_=self.compose(row,colors=labels)
        return im[:,:,:3]

    def render(self,row,colors=None,remove=None):
        im,_=self.compose(row,colors=colors,remove=remove)
        return torch.from_numpy(im[None]),torch.from_numpy(im[None,:,:,3,None]),{}
