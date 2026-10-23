"""2D Gaussian Splatting rasterizer for the existing 3DGRUT optimizer.

Each Gaussian is drawn as a flat disk. A long needle can no longer paint a
streak across the image the way the 3DGS rasterizer does.
"""
import torch
from gsplat import rasterization_2dgs


class Gsplat2DBackend:
    def __init__(self, conf):
        self.conf = conf
        self.timings = {}

    def build_acc(self, model, rebuild=True):
        pass

    def render(self, model, batch, train=False, frame_id=0):
        params = batch.intrinsics_OpenCVPinholeCameraModelParameters
        if params is None:
            raise ValueError('The car experiment requires pinhole intrinsics')
        device = model.positions.device
        focal = torch.as_tensor(params['focal_length'], device=device, dtype=torch.float32)
        center = torch.as_tensor(params['principal_point'], device=device, dtype=torch.float32)
        intrinsics = torch.eye(3, device=device)
        intrinsics[0, 0] = focal[0]
        intrinsics[1, 1] = focal[1]
        intrinsics[0, 2] = center[0]
        intrinsics[1, 2] = center[1]
        height, width = batch.rays_dir.shape[1:3]
        rendered, alpha, normals, surface_normals, distort, median, info = rasterization_2dgs(
            means=model.get_positions(), quats=model.get_rotation(), scales=model.get_scale(),
            opacities=model.get_density().flatten(), colors=model.get_features().reshape(model.num_gaussians, -1, 3),
            viewmats=torch.linalg.inv(batch.T_to_world).contiguous(), Ks=intrinsics[None],
            width=width, height=height, sh_degree=int(model.n_active_features), packed=False,
            near_plane=0.01, render_mode='RGB+ED', distloss=False)
        if surface_normals is None:
            raise RuntimeError('2DGS did not return surface normals')
        rgb, alpha = model.background(batch.T_to_world, batch.rays_dir, rendered[..., :3], alpha, train)
        radii = info['radii']
        visible = radii > 0
        if visible.ndim == 3:
            visible = visible.any(-1)
        if visible.ndim == 2:
            visible = visible.any(0)
        return dict(pred_rgb=rgb, pred_opacity=alpha, pred_dist=median, pred_normals=normals,
                    surf_normals=surface_normals, pred_distort=distort,
                    mog_visibility=visible.reshape(-1, 1))


def install():
    import threedgut_tracer
    threedgut_tracer.Tracer = Gsplat2DBackend
