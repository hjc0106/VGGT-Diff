"""Fit full-video SfM into Omega using real background pixel observations.

One common Sim3 is fitted to six original real cameras; spatial holdout pixels
never fit the transform. No generated RGB and no independent camera adjustment.
"""
from pathlib import Path
import importlib.util
import sys
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import vggtdiff_real_front_anchor_calibration as anchors
from vggtdiff_white_car import BASE,REPO,save

ROOT=REPO/'pipeline_exp/realworld/scene-1-nofull/3dgs_wcar'
OUT=anchors.OUT/'real_calibration/sfm_bridge'


def load_colmap():
    spec=importlib.util.spec_from_file_location('anchor_colmap_io',REPO/'thirdparty/map-anything/mapanything/utils/colmap.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def project(points,w,k):
    camera=points@w[:3,:3].T+w[:3,3];uv=camera@k.T
    return uv[:,:2]/np.maximum(uv[:,2:],1e-9)


def fit_bridge():
    cv2.setNumThreads(4);OUT.mkdir(parents=True,exist_ok=True)
    c=load_colmap();images=c.read_images_binary(ROOT/'colmap/sparse/0/images.bin');byname={r.name:r for r in images.values()}
    points=c.read_points3d_binary(ROOT/'colmap/sparse/0/points3D.bin')
    meta=anchors.gt.read(ROOT/'frames_manifest.json')['images'];omega=anchors.gt.read(anchors.gt.REAL/'camera_estimate.json')
    coverage=anchors.gt.read(BASE/'realvideo_batch08_coverage_v1/sfm_coverage.json')['alignment']
    s0=1/coverage['scale'];r0=np.asarray(coverage['rotation']).T;t0=-s0*r0@np.asarray(coverage['translation'])
    x0=np.r_[np.log(s0),Rotation.from_matrix(r0).as_rotvec(),t0]
    sift=cv2.SIFT_create(nfeatures=32000,contrastThreshold=.02)
    cache=OUT/'pixel_correspondences.npz'
    if cache.exists():
        with np.load(cache) as d:xyz,uv,view,holdout=[d[k] for k in ['xyz_sfm','uv1280','view','holdout']]
    else:
        xyzs=[];uvs=[];views=[]
        with np.load(anchors.CACHE) as d:
            for i,row in enumerate(omega['sources']):
                source=cv2.resize(cv2.imread(str(anchors.gt.e.INPUT/row['full_image'])),(1280,720))
                mask=anchors.gt.background_mask(source.shape,np.asarray(anchors.gt.e.POLYGONS[i])*4/3)
                skp,sd=sift.detectAndCompute(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY),mask)
                des=d[f'des{i}'];source_matches=cv2.BFMatcher(cv2.NORM_L2).match(des,sd)
                assert max(m.distance for m in source_matches)<1e-5,'SIFT cache and real-source pixels differ'
                source_xy=np.array([skp[m.trainIdx].pt for m in source_matches])
                nearest=min(meta,key=lambda r:abs(r['video_frame_index']-row['source_frame_zero_based']))
                im=byname[nearest['image']]
                target=cv2.resize(cv2.imread(str(ROOT/'images'/im.name)),(1280,720))
                tkp,td=sift.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY),mask)
                flann=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=128))
                cv2.setRNGSeed(20261022+i);matches=flann.knnMatch(des,td,k=2)
                valid=im.point3D_ids>=0;xy_sfm=im.xys[valid];ids=im.point3D_ids[valid]
                tree=cKDTree(xy_sfm);best={}
                for pair in matches:
                    if len(pair)!=2:continue
                    a,b=pair
                    if a.distance>=.75*b.distance:continue
                    distance,near=tree.query(np.array(tkp[a.trainIdx].pt)*3)
                    if distance>2.5:continue
                    pid=int(ids[near])
                    if pid not in points or (pid in best and a.distance>=best[pid][0]):continue
                    best[pid]=(a.distance,source_xy[a.queryIdx],points[pid].xyz)
                for value in best.values():xyzs.append(value[2]);uvs.append(value[1]);views.append(i)
                print('BRIDGE_REAL_PIXELS',i,nearest['image'],len(best),flush=True)
        xyz,uv,view=np.array(xyzs),np.array(uvs),np.array(views,int)
        holdout=((uv[:,0]//80+7*(uv[:,1]//80)).astype(int)%5)==0
        np.savez_compressed(cache,xyz_sfm=xyz,uv1280=uv,view=view,holdout=holdout)
    poses=np.array(omega['source_w2c']);ks=np.array(omega['source_intrinsics']);ks[:,:2]/=3
    def transform(x):return np.exp(x[0])*(xyz@Rotation.from_rotvec(x[1:4]).as_matrix().T)+x[4:]
    def residual(x,selected):
        p=transform(x);pred=np.empty_like(uv)
        for i in range(6):
            keep=view==i;pred[keep]=project(p[keep],poses[i],ks[i])
        return (pred-uv)[selected].ravel()
    train=~holdout
    result=least_squares(residual,x0,args=(train,),loss='soft_l1',f_scale=2,max_nfev=300,x_scale='jac')
    errors=np.linalg.norm(residual(result.x,np.ones(len(uv),bool)).reshape(-1,2),axis=1)
    inliers=train&(errors<4)
    result=least_squares(residual,result.x,args=(inliers,),loss='soft_l1',f_scale=1,max_nfev=300,x_scale='jac')
    errors=np.linalg.norm(residual(result.x,np.ones(len(uv),bool)).reshape(-1,2),axis=1)
    r=Rotation.from_rotvec(result.x[1:4]).as_matrix();s=float(np.exp(result.x[0]));t=result.x[4:]
    per_view=[]
    for i in range(6):
        h=(view==i)&holdout;tr=(view==i)&train
        per_view.append(dict(view=i,train_matches=int(tr.sum()),heldout_matches=int(h.sum()),
            train_median_px=float(np.median(errors[tr])),heldout_median_px=float(np.median(errors[h])) if h.any() else None,
            heldout_p90_px=float(np.percentile(errors[h],90)) if h.any() else None))
    payload=dict(scale_sfm_to_omega=s,rotation_sfm_to_omega=r.tolist(),translation_sfm_to_omega=t.tolist(),
        n_correspondences=len(uv),train_matches=int(train.sum()),heldout_matches=int(holdout.sum()),
        train_median_1280_px=float(np.median(errors[train])),heldout_median_1280_px=float(np.median(errors[holdout])),
        heldout_p90_1280_px=float(np.percentile(errors[holdout],90)),per_view=per_view,
        camera_status='Estimated alignment, not calibrated GT',
        method='SfM landmark IDs joined at2.5px to matching CV-SIFT pixels in nearest registered frames; fit one Sim3 against six original Omega real cameras, with spatial heldout pixels excluded from fitting.')
    save(OUT/'alignment.json',payload);print('BRIDGE_RESULT',payload,flush=True)
    return payload

if __name__=='__main__':fit_bridge()
