"""Continue batch6's final camera around the car without a distance/K jump.

Prepare only: --prepare-only. Otherwise generate eight batch7 views using the
same six generated references, checkpoint, seed and steps as the near-orbit run.
Historical scripts, images and experiment manifests are left untouched.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import vggtdiff_generated_batch_chain as chain

GEOMETRY=chain.BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json'
DEFAULT_OUTPUT=chain.BASE/'vggtdiff_batch06gen_to_batch07_continuous_v1'


def continuous_orbit(anchor_w2c,anchor_k,anchor_angle,center,up,angles):
    """Rigidly rotate the whole anchor camera about the center/up axis.

Increasing our vehicle azimuth means a negative right-handed rotation about up.
Rotating position and orientation together preserves distance, height, roll and
center projection, and reproduces the anchor camera exactly at its own angle.
"""
    center,up=np.asarray(center,dtype=float),np.asarray(up,dtype=float)
    w,k=np.asarray(anchor_w2c,dtype=float),np.asarray(anchor_k,dtype=float)
    if center.shape!=(3,) or up.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(up).all():
        raise ValueError('Invalid center/up')
    if np.linalg.norm(up)<1e-8: raise ValueError('Up axis is zero')
    up=up/np.linalg.norm(up)
    if w.shape!=(4,4) or k.shape!=(3,3) or not np.isfinite(w).all() or not np.isfinite(k).all():
        raise ValueError('Invalid anchor camera')
    if not np.allclose(w[3],[0,0,0,1]) or not np.allclose(w[:3,:3]@w[:3,:3].T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(w[:3,:3]),1,atol=1e-6):
        raise ValueError('Anchor is not a rigid W2C camera')
    if k[0,0]<=0 or k[1,1]<=0: raise ValueError('Invalid focal length')
    c2w=np.linalg.inv(w);offset=c2w[:3,3]-center
    if np.linalg.norm(offset)<1e-8: raise ValueError('Anchor coincides with vehicle center')
    angles=np.asarray(angles,dtype=float)
    if angles.ndim!=1 or not np.isfinite(angles).all() or not np.isfinite(anchor_angle): raise ValueError('Invalid angles')
    targets=[]
    for angle in angles:
        rotate=Rotation.from_rotvec(-np.deg2rad(angle-anchor_angle)*up).as_matrix()
        camera=np.eye(4);camera[:3,:3]=rotate@c2w[:3,:3]
        camera[:3,3]=center+rotate@offset
        targets.append(np.linalg.inv(camera).tolist())
    return dict(target_w2c=targets,target_intrinsics=[k.tolist() for _ in angles],
        target_angles_deg=angles.tolist(),intrinsics_image_size=[832,480],camera_convention='OpenCV W2C',
        note='Rigid continuation of the actual batch6 final camera; fixed center/up, radius, height, K and camera roll.')


def write_frozen_json(path,value):
    if path.exists() and chain.read(path)!=value:
        raise ValueError(f'Frozen camera design differs: {path}; choose a new --output')
    chain.save(path,value)


def prepare(args):
    source=args.source_run.resolve()
    source_trajectory=(args.source_trajectory or source.parent/'trajectory.json').resolve()
    original=chain.read(source_trajectory)
    if args.anchor_index not in args.reference_indices: raise ValueError('The anchor must be one of the six references')
    if not 0<=args.anchor_index<len(original['target_w2c']): raise ValueError('Invalid anchor index')
    anchor_angle=original['target_angles_deg'][args.anchor_index]
    indices=np.arange(48,56)
    if anchor_angle>=270: raise ValueError('Batch7 continuation expects an anchor before 270 degrees')
    output=args.output.resolve()
    if output in [source,source.parent,chain.BASE.resolve()] or source.is_relative_to(output): raise ValueError('Output overlaps source')
    g=chain.read(GEOMETRY);center=np.asarray(g['center']);up=np.asarray(g['up']);up=up/np.linalg.norm(up)
    w=np.asarray(original['target_w2c'][args.anchor_index]);k=np.asarray(original['target_intrinsics'][args.anchor_index])
    orbit=continuous_orbit(w,k,anchor_angle,center,up,np.arange(64)*5.625)
    anchor_check=continuous_orbit(w,k,anchor_angle,center,up,[anchor_angle])
    assert np.allclose(anchor_check['target_w2c'][0],w,atol=1e-12)
    offset=np.linalg.inv(w)[:3,3]-center;height=float(offset@up)
    radius=float(np.linalg.norm(offset-height*up));distance=float(np.linalg.norm(offset))
    center_cam=w@np.r_[center,1]
    if center_cam[2]<=0: raise ValueError('Vehicle center is behind anchor camera')
    checks=[]
    for i in indices:
        tw=np.asarray(orbit['target_w2c'][i]);delta=np.linalg.inv(tw)[:3,3]-center
        h=float(delta@up);r=float(np.linalg.norm(delta-h*up))
        cc=tw@np.r_[center,1]
        azimuth=float(np.degrees(np.arctan2(-delta@np.asarray(g['left']),delta@np.asarray(g['front'])))%360)
        assert abs(((azimuth-orbit['target_angles_deg'][i]+180)%360)-180)<1e-4
        assert abs(r-radius)<1e-10 and abs(h-height)<1e-10
        assert np.allclose(cc,center_cam,atol=1e-10)
        assert np.array_equal(orbit['target_intrinsics'][i],k)
        checks.append(dict(index=int(i),angle_deg=orbit['target_angles_deg'][i],
            measured_azimuth_deg=azimuth,horizontal_radius=r,height=h,distance=float(np.linalg.norm(delta)),
            fx=float(k[0,0]),fy=float(k[1,1]),center_depth=float(cc[2]),
            central_scale_ratio_to_anchor=float(center_cam[2]/cc[2])))
    audit=dict(state='passed',anchor_local_index=args.anchor_index,anchor_angle_deg=anchor_angle,
        anchor_w2c=w.tolist(),anchor_intrinsics=k.tolist(),center=center.tolist(),up=up.tolist(),
        horizontal_radius=radius,height=height,distance=distance,
        boundary_angle_change_deg=270-anchor_angle,targets=checks,
        method='Rotate anchor camera position and orientation together about center/up; preserve roll as well as distance and K.',
        units='Existing relative world units, not independently calibrated meters',
        checks=['anchor reconstructed exactly','radius/height/distance fixed','intrinsics identical',
                'correct vehicle azimuth','center camera coordinates fixed','reference images and poses unchanged'])
    # Distinct file required by the historical chain runner, which slices the 64-view orbit.
    orbit_path=output/'camera_design/target_orbit64.json'
    write_frozen_json(orbit_path,orbit)
    write_frozen_json(output/'camera_design/audit.json',audit)
    sources=[Path(__file__).resolve(),Path(chain.__file__).resolve(),GEOMETRY,source_trajectory,orbit_path,output/'camera_design/audit.json']
    design_manifest=dict(files=[dict(path=str(p),sha256=chain.digest(p)) for p in sources])
    write_frozen_json(output/'camera_design/manifest.json',design_manifest)
    namespace=argparse.Namespace(source_run=source,source_trajectory=source_trajectory,
        reference_indices=args.reference_indices,orbit_trajectory=orbit_path,batch=7,output=output,
        steps=args.steps,seed=args.seed,vram_limit_gib=args.vram_limit_gib)
    prepared=chain.prepare(namespace)
    m,t=chain.validate(prepared)
    for j,i in enumerate(args.reference_indices):
        assert np.array_equal(t['source_w2c'][j],original['target_w2c'][i])
        assert np.array_equal(t['source_intrinsics'][j],original['target_intrinsics'][i])
    for item in design_manifest['files']:
        assert chain.digest(Path(item['path']))==item['sha256']
    print('CONTINUITY_AUDIT_PASSED',dict(radius=radius,height=height,distance=distance,
        fx=float(k[0,0]),fy=float(k[1,1]),boundary_angle_change_deg=270-anchor_angle),flush=True)
    return prepared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,default=chain.BASE/'crossview_geometry_matched_gt_v1/seed_20261028')
    p.add_argument('--source-trajectory',type=Path)
    p.add_argument('--reference-indices',type=int,nargs=6,default=[0,1,3,4,6,7])
    p.add_argument('--anchor-index',type=int,default=7)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--vram-limit-gib',type=float,default=20)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();output=prepare(args)
    if not args.prepare_only: chain.infer(output)

if __name__=='__main__':main()
