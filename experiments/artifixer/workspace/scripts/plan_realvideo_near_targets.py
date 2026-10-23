"""Refine unseen-side cameras using the observed scene's occlusion proxy."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import vggtdiff_realvideo_unseen as experiment

PARENT=experiment.OUT
OUT=PARENT.with_name('vggtdiff_realvideo_unseen_right_near_v2')


def main():
    if (OUT/'inference_manifest.json').exists():raise FileExistsError('Refinement already frozen')
    g=json.loads((PARENT/'geometry_check.json').read_text())
    m=json.loads((PARENT/'camera_estimate.json').read_text())
    trajectory=json.loads((PARENT/'trajectory.json').read_text())
    with np.load(PARENT/'omega_dense.npz') as d:
        xyz=d['world'];confidence=d['confidence']
    center=np.array(g['center']);axes=np.stack([g['front'],g['left'],g['up']],axis=1)
    origin=np.array(g['ground_origin']);lo,hi=np.array(g['observed_car_bounds_local'])
    scene=[];cars=[]
    for i in range(6):
        points=xyz[i,::2,::2];conf=confidence[i,::2,::2]
        valid=np.isfinite(points).all(-1)&(conf>=np.quantile(conf,.5))
        scene.append(points[valid])
        h,w=confidence[i].shape;mask=np.zeros((h,w),np.uint8)
        poly=np.array(experiment.POLYGONS[i])*4
        poly=(np.c_[poly,np.ones(len(poly))]@np.linalg.inv(np.array(m['processed_to_original'][i])).T)[:,:2]
        cv2.fillPoly(mask,[np.rint(poly).astype(np.int32)],255)
        valid=(mask>0)&(confidence[i]>=np.quantile(confidence[i][mask>0],.3))
        cars.append(xyz[i][valid][::2])
    scene=np.concatenate(scene);cars=np.concatenate(cars)
    local=(scene-origin)@axes
    expansion=np.array([.1,.3,.15])*g['observed_car_dimensions'][0]
    inside=((local>lo-expansion)&(local<hi+expansion)).all(1)
    background=scene[~inside]
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@axes.T+origin
    original_radius=g['target_radius'];original_k=np.array(trajectory['target_intrinsics'][0]);height=g['target_height_relative_to_body_center']
    options=[]
    for ratio in [1.,.8,.65,.5,.4]:
        radius=original_radius*ratio;k=original_k.copy();k[0,0]*=ratio;k[1,1]*=ratio
        for _ in range(30):
            ws=[];positions=[];bounds=[]
            for angle in g['target_angles_deg']:
                t=np.deg2rad(angle);p=center+radius*(np.cos(t)*axes[:,0]-np.sin(t)*axes[:,1])+height*axes[:,2]
                w=experiment.look_at(p,center,axes[:,2]);uv,z=experiment.project(corners,w,k)
                assert z.min()>0
                ws.append(w);positions.append(p);bounds.append([*uv.min(0),*uv.max(0)])
            if all(b[0]>45 and b[2]<787 and b[1]>35 and b[3]<445 for b in bounds):break
            k[0,0]*=.97;k[1,1]*=.97
        scores=[];distances=[]
        for p,w in zip(positions,ws):
            small=k.copy();small[:2]/=2
            uv,z=experiment.project(background,w,small);pixels=np.rint(uv).astype(np.int64)
            valid=(z>0)&(pixels[:,0]>=0)&(pixels[:,0]<416)&(pixels[:,1]>=0)&(pixels[:,1]<240)
            depth=np.full(416*240,np.inf,np.float32)
            np.minimum.at(depth,pixels[valid,1]*416+pixels[valid,0],z[valid].astype(np.float32))
            depth=cv2.erode(depth.reshape(240,416),np.ones((3,3),np.uint8))
            uv,z=experiment.project(cars,w,small);pixels=np.rint(uv).astype(np.int64)
            valid=(z>0)&(pixels[:,0]>=0)&(pixels[:,0]<416)&(pixels[:,1]>=0)&(pixels[:,1]<240)
            foreground_depth=depth[pixels[valid,1],pixels[valid,0]]
            scores.append(float((foreground_depth<z[valid]-.03).mean()))
            distances.append(float(np.linalg.norm(background-p,axis=1).min()))
        options.append(dict(radius_ratio=ratio,radius=radius,height_relative_to_center=height,
            target_intrinsics=k.tolist(),target_positions=np.array(positions).tolist(),target_w2c=np.array(ws).tolist(),
            target_bounds=bounds,known_scene_occlusion_proxy_by_view=scores,
            mean_occlusion_proxy=float(np.mean(scores)),minimum_distance_to_known_scene_by_view=distances))
    # The partial point proxy underestimates the strong vegetation seen in actual RGB.
    # Use a preselected half-distance test, rather than claiming proxy visibility is GT.
    chosen=next(o for o in options if o['radius_ratio']==.5)
    assert min(chosen['minimum_distance_to_known_scene_by_view'])>.01
    OUT.mkdir(exist_ok=True)
    for name in ['camera_estimate.json','match_checks.json','sfm_crosscheck.json','crop_alignment_check.json',
                 'crop_alignment_check_before_rectification.json','source_preprocessing.json','runtime_versions.json',
                 'model_source_contact.jpg','selected_sources_preview.jpg']:
        shutil.copyfile(PARENT/name,OUT/name)
    shutil.copytree(PARENT/'source_views',OUT/'source_views',dirs_exist_ok=True)
    g.update(parent_experiment=str(PARENT),target_radius=chosen['radius'],target_positions=chosen['target_positions'],
        target_bounds_model_pixels=chosen['target_bounds'],target_radius_refinement='Move nearer to avoid foreground vegetation; scale focal lengths to retain framing; retain camera height')
    trajectory['target_w2c']=chosen['target_w2c'];trajectory['target_intrinsics']=[chosen['target_intrinsics']]*5
    experiment.save(OUT/'geometry_check.json',g);experiment.save(OUT/'trajectory.json',trajectory)
    experiment.save(OUT/'target_refinement.json',dict(parent_experiment=str(PARENT),options=options,chosen=chosen,
        policy='Controlled half-distance camera test, with focal lengths scaled to retain framing and physical camera height unchanged; source RGB/poses, angles, seed and steps stay fixed',
        proxy_audit='Original RGB shows strong vegetation obstruction but the observed full-scene point proxy reports <3% per view; it is too incomplete/confidence-filtered to guarantee target visibility',
        limitation='Proxy from partial Omega scene points, not complete geometry, visibility truth, or a guarantee against collision; generated occlusion can differ'))
    for i in range(6):assert experiment.digest(OUT/'source_views'/f'{i:02d}.png')==experiment.digest(PARENT/'source_views'/f'{i:02d}.png')
    shutil.copyfile(Path(__file__),OUT/'planning_script_snapshot.py')
    shutil.copyfile(experiment.REPO/'scripts/vggtdiff_realvideo_unseen.py',OUT/'inference_script_snapshot.py')
    print(json.dumps(dict(options=[{k:o[k] for k in ['radius_ratio','mean_occlusion_proxy','known_scene_occlusion_proxy_by_view']} for o in options],chosen_ratio=chosen['radius_ratio'],output=str(OUT)),indent=2))


if __name__=='__main__':main()
