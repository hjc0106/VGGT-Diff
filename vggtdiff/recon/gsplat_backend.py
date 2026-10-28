"""gsplat rasterizer used by the E1 3DGS trainer in place of the 3DGRUT tracer."""

from __future__ import annotations

import torch


def install() -> None:
    import threedgut_tracer

    threedgut_tracer.Tracer = GsplatBackend


class GsplatBackend:
    def __init__(self, conf):
        self.conf = conf
        self.timings = {}

    def build_acc(self, model, rebuild=True):
        del model, rebuild

    def render(self, model, batch, train=False, frame_id=0):
        from gsplat import rasterization

        del frame_id
        params = batch.intrinsics_OpenCVPinholeCameraModelParameters
        if params is None:
            raise ValueError("E1 reconstruction requires pinhole intrinsics")
        device = model.positions.device
        focal = torch.as_tensor(params["focal_length"], device=device, dtype=torch.float32)
        principal = torch.as_tensor(params["principal_point"], device=device, dtype=torch.float32)
        intrinsics = torch.eye(3, device=device)
        intrinsics[0, 0] = focal[0]
        intrinsics[1, 1] = focal[1]
        intrinsics[0, 2] = principal[0]
        intrinsics[1, 2] = principal[1]
        height, width = batch.rays_dir.shape[1:3]
        features = model.get_features().reshape(model.num_gaussians, -1, 3)
        rgbd, alpha, info = rasterization(
            means=model.get_positions(),
            quats=model.get_rotation(),
            scales=model.get_scale(),
            opacities=model.get_density().flatten(),
            colors=features,
            viewmats=torch.linalg.inv(batch.T_to_world).contiguous(),
            Ks=intrinsics[None],
            width=width,
            height=height,
            sh_degree=int(model.n_active_features),
            packed=False,
            near_plane=0.001,
            render_mode="RGB+ED",
        )
        rgb = rgbd[:, :, :, :3]
        rgb, alpha = model.background(batch.T_to_world, batch.rays_dir, rgb, alpha, train)
        visible = (info["radii"] > 0).any(0)[:, None]
        return dict(
            pred_rgb=rgb,
            pred_opacity=alpha,
            pred_dist=rgbd[:, :, :, 3:4],
            mog_visibility=visible,
        )
