"""Real video -> checked source cameras -> custom unseen-side NVS experiment."""
from __future__ import annotations
from experiment_paths import checkpoint_path, omega_path, base_model_path
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from vggtdiff_white_car import BASE, REPO, digest, save, sheet

INPUT=BASE/'vggtdiff_realvideo_refs_v1'
OUT=BASE/'vggtdiff_realvideo_unseen_right_v1'
SELECTED=[1,3,4,5,6,8]
OMEGA=Path(str(omega_path()))
POLYGONS=[
 [(269,270),(280,249),(307,242),(325,227),(377,222),(407,230),(425,244),(430,263),(420,276),(345,292),(284,290)],
 [(332,272),(344,259),(370,255),(398,241),(448,239),(482,246),(516,261),(523,282),(508,295),(384,305),(338,296)],
 [(407,260),(425,247),(450,244),(478,232),(525,229),(556,236),(597,250),(605,275),(589,288),(444,294),(409,283)],
 [(474,263),(493,250),(526,248),(553,233),(608,230),(650,240),(687,256),(693,278),(676,297),(541,299),(479,285)],
 [(501,270),(518,250),(548,247),(573,233),(630,229),(688,242),(739,267),(750,292),(732,310),(643,320),(512,289)],
 [(541,281),(552,259),(582,250),(613,232),(671,229),(738,249),(810,279),(826,310),(818,337),(751,350),(655,343),(547,306)],
]


def unit(x):
    return x/max(np.linalg.norm(x),1e-12)


def contact(paths,labels,dest,columns=3):
    canvas=Image.new('RGB',(columns*416,((len(paths)+columns-1)//columns)*272),'#181818')
    draw=ImageDraw.Draw(canvas)
    for j,(path,label) in enumerate(zip(paths,labels)):
        x=j%columns*416;y=j//columns*272
        with Image.open(path) as im:
            tile=ImageOps.pad(im.convert('RGB'),(416,240),method=Image.Resampling.LANCZOS,color='#101010')
        canvas.paste(tile,(x,y+32));draw.text((x+8,y+9),label,fill='white')
    canvas.save(dest,quality=95)


def check_crop():
    import torch
    from vggtdiff.omega import OmegaRuntime,_similarity_transform
    m=json.loads((OUT/'camera_estimate.json').read_text())
    trajectory=json.loads((OUT/'trajectory.json').read_text())
    omega=OmegaRuntime(str(OMEGA),torch.device('cuda'))
    images=[Image.open(OUT/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
    predicted,k=omega.estimate_cameras(images)
    raw=np.asarray(trajectory['source_w2c']);pc=np.linalg.inv(predicted)[:,:3,3];rc=np.linalg.inv(raw)[:,:3,3]
    scale,rotation,translation=_similarity_transform(pc,rc)
    errors=np.linalg.norm(scale*(pc@rotation.T)+translation-rc,axis=1)
    r_aligned=predicted[:,:3,:3]@rotation.T
    delta=r_aligned@raw[:,:3,:3].transpose(0,2,1)
    degrees=np.rad2deg(np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1)))
    k_expected=np.asarray(trajectory['source_intrinsics'])
    result=dict(center_alignment_rmse_relative_to_original_baseline=float(np.sqrt(np.mean(errors**2))/m['median_baseline']),
        rotation_error_degrees=degrees.tolist(),median_rotation_error_degrees=float(np.median(degrees)),
        focal_ratio_to_expected=(k[:,[0,1],[0,1]]/k_expected[:,[0,1],[0,1]]).tolist(),
        source_predicted_w2c=predicted.tolist(),source_predicted_intrinsics=k.tolist(),
        note='Independent Omega estimate on the actual final model RGB inputs; compared to their calibrated supplied cameras')
    save(OUT/'crop_alignment_check.json',result)
    contact([OUT/'source_views'/f'{i:02d}.png' for i in range(6)],
        [f'real source {i:02d} | candidate {c:02d}' for i,c in enumerate(SELECTED)],OUT/'model_source_contact.jpg')
    contact([INPUT/x['crop_image'] for x in m['sources']],
        [f"candidate {x['candidate']:02d} | {x['pts_seconds']:.2f}s" for x in m['sources']],OUT/'selected_sources_preview.jpg')
    print(json.dumps(result,indent=2),flush=True)


def rectify():
    import cv2
    m=json.loads((OUT/'camera_estimate.json').read_text())
    g=json.loads((OUT/'geometry_check.json').read_text())
    t=json.loads((OUT/'trajectory.json').read_text())
    if (OUT/'source_preprocessing.json').exists():raise FileExistsError('Sources already rectified')
    if (OUT/'crop_alignment_check.json').exists():
        (OUT/'crop_alignment_check.json').rename(OUT/'crop_alignment_check_before_rectification.json')
    center=np.array(g['center']);up=np.array(g['up']);front=np.array(g['front']);left=np.array(g['left'])
    axes=np.stack([front,left,up],axis=1);lo,hi=np.array(g['observed_car_bounds_local'])
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@axes.T+np.array(g['ground_origin'])
    records=[];ws=[];ks=[]
    for i,item in enumerate(m['sources']):
        old=np.array(m['source_w2c'][i]);ko=np.array(m['source_intrinsics'][i]);position=np.linalg.inv(old)[:3,3]
        new=look_at(position,center,up)
        initial=np.array(t['source_intrinsics'][i]);kn=np.array([[initial[0,0],0,415.5],[0,initial[1,1],239.5],[0,0,1.]])
        for _ in range(30):
            uv,z=project(corners,new,kn)
            if z.min()>0 and uv[:,0].min()>35 and uv[:,0].max()<797 and uv[:,1].min()>30 and uv[:,1].max()<450:break
            kn[0,0]*=.97;kn[1,1]*=.97
        else:raise ValueError('Source rectification framing failed')
        homography=kn@new[:3,:3]@np.linalg.inv(old[:3,:3])@np.linalg.inv(ko)
        original=cv2.imread(str(INPUT/item['full_image']))
        image=cv2.warpPerspective(original,homography,(832,480),flags=cv2.INTER_CUBIC)
        valid=cv2.warpPerspective(np.full(original.shape[:2],255,np.uint8),homography,(832,480),flags=cv2.INTER_NEAREST)
        assert (valid==255).mean()>.999,'Virtual reference exits real image'
        # Verify the image homography and the new camera encode exactly the same rays.
        before,_=project(corners,old,ko);after,_=project(corners,new,kn)
        mapped=np.c_[before,np.ones(len(before))]@homography.T;mapped=mapped[:,:2]/mapped[:,2:]
        error=float(np.abs(mapped-after).max());assert error<1e-8
        cv2.imwrite(str(OUT/'source_views'/f'{i:02d}.png'),image)
        records.append(dict(candidate=item['candidate'],homography=homography.tolist(),
            original_w2c=old.tolist(),virtual_w2c=new.tolist(),virtual_intrinsics=kn.tolist(),
            homography_projection_error=error,valid_fraction=float((valid==255).mean()),
            projected_car_bounds=[*uv.min(0),*uv.max(0)]))
        ws.append(new.tolist());ks.append(kn.tolist())
    t['source_w2c']=ws;t['source_intrinsics']=ks
    save(OUT/'trajectory.json',t)
    save(OUT/'source_preprocessing.json',dict(mode='Calibrated rotational homography to centered virtual pinhole cameras',
        source='Original decoded video PNGs; no generated pixels; camera centers unchanged; source rotation and K updated with image warp',
        prior_crop_boxes_role='Only initialize field of view; direct off-axis crops are not used for generation',frames=records))
    contact([OUT/'source_views'/f'{i:02d}.png' for i in range(6)],
        [f'perspective-corrected real source | {c:02d}' for c in SELECTED],OUT/'model_source_contact.jpg')


def validate_manifest(manifest):
    for entry in manifest['frozen_files']:
        assert digest(Path(entry['path']))==entry['sha256'],f"Input changed: {entry['path']}"


def infer():
    import torch
    import imageio.v2 as imageio
    from vggtdiff import VGGTDiff

    m=json.loads((OUT/'camera_estimate.json').read_text())
    geometry=json.loads((OUT/'geometry_check.json').read_text())
    crop_check=json.loads((OUT/'crop_alignment_check.json').read_text())
    assert geometry['all_targets_on_right'] and geometry['median_car_sampson_px']<3
    assert crop_check['center_alignment_rmse_relative_to_original_baseline']<.03
    assert max(crop_check['rotation_error_degrees'])<2
    trajectory=json.loads((OUT/'trajectory.json').read_text())
    checkpoint=Path(str(checkpoint_path()))
    config=json.loads(checkpoint.with_name('config.json').read_text());assert config['model_family']=='nvs'
    base=Path(str(base_model_path()))
    paths=[INPUT/item['full_image'] for item in m['sources']]
    for p,item in zip(paths,m['sources']):assert digest(p)==item['full_sha256']
    paths += [OUT/'source_views'/f'{i:02d}.png' for i in range(6)]
    paths += [OUT/x for x in ['camera_estimate.json','geometry_check.json','trajectory.json','crop_alignment_check.json','match_checks.json','sfm_crosscheck.json','source_preprocessing.json']]
    paths.append(checkpoint.with_name('config.json'))
    manifest=dict(source='Real video only; no PLY-rendered or synthetic RGB conditioning',
        selected_candidates=SELECTED,source_video=json.loads((INPUT/'candidates_manifest.json').read_text())['video'],
        target_angles_deg=geometry['target_angles_deg'],target_count=5,seed=20261020,steps=50,
        resolution=[832,480],training_mode='nvs',dit_eval=True,vram_limit_gib=20,
        checkpoint=str(checkpoint),omega=str(OMEGA),base=str(base),
        cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(),
        frozen_files=[dict(path=str(p),sha256=digest(p)) for p in paths],
        weights=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in [checkpoint,OMEGA]])
    path=OUT/'inference_manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:save(path,manifest)
    validate_manifest(manifest)
    run=OUT/'pilot';run.mkdir(exist_ok=True)
    (run/'raw').mkdir(exist_ok=True)
    save(OUT/'status.json',dict(state='loading_diffusion',target_count=5,completed_frames=0))
    model=VGGTDiff(str(checkpoint),str(OMEGA),base_model_dir=str(base),device='cuda',
        vram_limit_gib=20,training_mode='nvs',load_full_checkpoint_directly=True)
    assert not model.pipe.dit.training
    images=[Image.open(OUT/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
    w=np.array(trajectory['source_w2c']+trajectory['target_w2c'],dtype=np.float32)
    k=np.array(trajectory['source_intrinsics']+trajectory['target_intrinsics'],dtype=np.float32)
    assert w.shape==(11,4,4) and k.shape==(11,3,3)
    save(OUT/'status.json',dict(state='generating',target_count=5,completed_frames=0,seed=20261020,steps=50))
    print('START_GENERATE_FIVE_RIGHT_SIDE_VIEWS',flush=True)
    torch.cuda.reset_peak_memory_stats();start=time.time()
    outputs=model.generate(images,w,k,height=480,width=832,steps=50,seed=20261020)
    assert len(outputs)==5
    records=[];video=[]
    for i,(angle,im) in enumerate(zip(geometry['target_angles_deg'],outputs)):
        assert im.size==(832,480)
        dest=run/'raw'/f'{i:02d}.png';im.save(dest)
        panel=Image.new('RGB',(832,512),'#181818');panel.paste(im,(0,32))
        ImageDraw.Draw(panel).text((12,10),f'Unseen RIGHT view | {angle:g} degrees | real-video references',fill='white')
        video.append(np.array(panel));records.append(dict(index=i,angle_degrees=angle,path=str(dest.relative_to(OUT)),sha256=digest(dest)))
    contact([run/'raw'/f'{i:02d}.png' for i in range(5)],
        [f'RIGHT {a:g} deg' for a in geometry['target_angles_deg']],run/'predictions_contact.jpg')
    imageio.mimsave(run/'prediction.mp4',video,fps=1,macro_block_size=16)
    result=dict(frames=records,elapsed_seconds=time.time()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,visual_review='pending')
    save(run/'result.json',result);validate_manifest(manifest)
    for entry in manifest['weights']:
        p=Path(entry['path']);assert p.stat().st_size==entry['size'] and p.stat().st_mtime_ns==entry['mtime_ns']
    save(OUT/'status.json',dict(state='completed_pending_visual_review',completed_frames=5,target_count=5))
    print('COMPLETED_FIVE_RIGHT_SIDE_VIEWS',flush=True)


def project(xyz,w,k):
    camera=np.asarray(xyz)@w[:3,:3].T+w[:3,3]
    xy=camera@k.T
    return xy[:,:2]/xy[:,2:],camera[:,2]


def look_at(position,center,up):
    z=unit(center-position);x=unit(np.cross(z,up));y=np.cross(z,x)
    w=np.eye(4);w[:3,:3]=np.stack([x,y,z]);w[:3,3]=-w[:3,:3]@position
    assert abs(np.linalg.det(w[:3,:3])-1)<1e-8
    return w


def match_checks(m,d):
    import cv2
    cv2.setNumThreads(4)
    keypoints=[];descriptors=[];masks=[];images=[]
    sift=cv2.SIFT_create(nfeatures=12000)
    for item,polygon in zip(m['sources'],POLYGONS):
        path=INPUT/item['full_image']
        im=cv2.imread(str(path));im=cv2.resize(im,(1280,720))
        mask=np.zeros(im.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.round(np.array(polygon)*4/3).astype(np.int32)],255)
        kp,desc=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
        keypoints.append(kp);descriptors.append(desc);masks.append(mask);images.append(im)
    w=np.asarray(m['source_w2c']);k=np.asarray(m['source_intrinsics']);k[:,:2]/=3
    records=[];matcher=cv2.BFMatcher()
    for i in range(5):
        j=i+1
        forward=matcher.knnMatch(descriptors[i],descriptors[j],k=2)
        reverse=matcher.knnMatch(descriptors[j],descriptors[i],k=2)
        rev={a.queryIdx:a.trainIdx for a,b in reverse if a.distance<.75*b.distance}
        matched=[a for a,b in forward if a.distance<.75*b.distance and rev.get(a.trainIdx)==a.queryIdx]
        p=np.array([keypoints[i][a.queryIdx].pt for a in matched]);q=np.array([keypoints[j][a.trainIdx].pt for a in matched])
        region=np.array([bool(masks[i][round(y),round(x)] and masks[j][round(v),round(u)]) for (x,y),(u,v) in zip(p,q)])
        relative=w[j]@np.linalg.inv(w[i]);r=relative[:3,:3];t=relative[:3,3]
        skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
        f=np.linalg.inv(k[j]).T@skew@r@np.linalg.inv(k[i])
        p1=np.c_[p,np.ones(len(p))];q1=np.c_[q,np.ones(len(q))]
        fp=p1@f.T;fq=q1@f
        error=np.abs(np.sum(q1*fp,axis=1))/np.sqrt((fp[:,:2]**2).sum(1)+(fq[:,:2]**2).sum(1))
        # Independent robust F tests distinguish bad matches from pose discrepancies.
        _,inliers=cv2.findFundamentalMat(p,q,cv2.USAC_MAGSAC,1.5,.999,10000)
        inliers=inliers.ravel().astype(bool)
        groups={}
        for name,selected in [('car',region&inliers),('background',~region&inliers)]:
            groups[name]=dict(matches=int(selected.sum()),median_sampson_px=float(np.median(error[selected])) if selected.any() else None,
                fraction_below_3px=float((error[selected]<3).mean()) if selected.any() else None)
        records.append(dict(source_pair=[m['selected_candidates'][i],m['selected_candidates'][j]],
            image_resolution=[1280,720],total_mutual_matches=len(matched),robust_inliers=int(inliers.sum()),groups=groups))
        valid=region&inliers
        if valid.sum()>0:
            # Test dense Omega geometry against independently matched pixels in another view.
            a=np.asarray(m['processed_to_original'][i]);uv=np.linalg.inv(a)@np.c_[p[valid]*3,np.ones(valid.sum())].T
            xx=np.rint(uv[0]).astype(int).clip(0,d['world'].shape[2]-1)
            yy=np.rint(uv[1]).astype(int).clip(0,d['world'].shape[1]-1)
            reprojected,depth=project(d['world'][i,yy,xx],w[j],k[j])
            errors=np.linalg.norm(reprojected-q[valid],axis=1)
            records[-1]['car_dense_transfer_median_px']=float(np.median(errors[depth>0]))
    save(OUT/'match_checks.json',dict(method='Mutual SIFT + independently robust F inliers; Omega-camera Sampson error; dense cross-view transfer',pairs=records))
    return records


def calibrate():
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=json.loads((OUT/'camera_estimate.json').read_text());d=np.load(OUT/'omega_dense.npz')
    checks=match_checks(m,d)
    car=[];ground=[];car_masks=[]
    for i,polygon in enumerate(POLYGONS):
        h,width=d['confidence'][i].shape;a=np.asarray(m['processed_to_original'][i])
        mask=np.zeros((h,width),np.uint8)
        poly=(np.linalg.inv(a)@np.c_[np.array(polygon)*4,np.ones(len(polygon))].T).T[:,:2]
        cv2.fillPoly(mask,[np.rint(poly).astype(np.int32)],255);car_masks.append(mask)
        confidence=d['confidence'][i];valid=(mask>0)&np.isfinite(d['world'][i]).all(-1)
        valid &= confidence>=np.quantile(confidence[valid],.30)
        car.append(d['world'][i][valid])
        box=np.array(polygon);x0=max(box[:,0].min()-30,0);x1=min(box[:,0].max()+40,959)
        y0=box[:,1].max()+25;y1=min(y0+80,539)
        yy,xx=np.mgrid[:h,:width];pixels=np.stack([xx,yy,np.ones_like(xx)],-1)@a.T/4
        road=(pixels[...,0]>x0)&(pixels[...,0]<x1)&(pixels[...,1]>y0)&(pixels[...,1]<y1)
        road &= confidence>=np.quantile(confidence[road],.30)
        ground.append(d['world'][i][road])
    ground=np.concatenate(ground);car=np.concatenate(car)
    centers=np.linalg.inv(np.asarray(m['source_w2c']))[:,:3,3]
    rng=np.random.default_rng(20261020);sample=ground[rng.choice(len(ground),min(10000,len(ground)),replace=False)]
    tolerance=m['median_baseline']*.015;best=np.zeros(len(sample),bool)
    for _ in range(800):
        p=sample[rng.choice(len(sample),3,replace=False)];normal=unit(np.cross(p[1]-p[0],p[2]-p[0]))
        residual=np.abs((sample-p[0])@normal);inliers=residual<tolerance
        if inliers.sum()>best.sum():best=inliers
    inlier_points=sample[best];origin=np.median(inlier_points,axis=0)
    _,_,v=np.linalg.svd(inlier_points-origin,full_matrices=False);up=v[-1]
    if np.dot(centers.mean(0)-origin,up)<0:up=-up
    # A visually annotated side view determines head/tail sign independently of PCA.
    a=np.asarray(m['processed_to_original'][2]);h,width=d['confidence'][2].shape
    yy,xx=np.mgrid[:h,:width];pixels=np.stack([xx,yy,np.ones_like(xx)],-1)@a.T/4
    def region_center(box):
        x0,y0,x1,y1=box
        region=(pixels[...,0]>x0)&(pixels[...,0]<x1)&(pixels[...,1]>y0)&(pixels[...,1]<y1)&(car_masks[2]>0)
        return np.median(d['world'][2][region],axis=0)
    front_hint=region_center([415,248,460,280]);rear_hint=region_center([550,245,593,280])
    front=front_hint-rear_hint;front=unit(front-up*np.dot(front,up));left=unit(np.cross(up,front))
    axes=np.stack([front,left,up],axis=1)
    local=(car-origin)@axes
    lo,hi=np.quantile(local,[.025,.975],axis=0)
    center_local=(lo+hi)/2
    center=origin+axes@center_local
    length=float(hi[0]-lo[0]);height=float(hi[2]-lo[2]);width_car=float(hi[1]-lo[1])
    assert best.mean()>.5,'Road plane is unreliable'
    source_local=(centers-center)@axes
    assert np.all(source_local[:,1]>0),'Estimated vehicle left/right contradict reference images'
    k_source=[];source_paths=[];crops=[]
    (OUT/'source_views').mkdir(exist_ok=True)
    for i,item in enumerate(m['sources']):
        box=item['preview_crop_xyxy'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        n=int(np.ceil(max((box[2]-box[0])/26,(box[3]-box[1])/15)))
        cw,ch=26*n,15*n;x0=int(round(cx-cw/2));y0=int(round(cy-ch/2))
        crop=[x0,y0,x0+cw,y0+ch];assert 0<=x0 and x0+cw<=3840 and 0<=y0 and y0+ch<=2160
        s=832/cw;pixel=np.array([[s,0,-s*x0],[0,s,-s*y0],[0,0,1.]])
        k_source.append((pixel@np.array(m['source_intrinsics'][i])).tolist());crops.append(dict(box=crop,pixel_transform=pixel.tolist()))
        with Image.open(INPUT/item['full_image']) as im:processed=im.crop(crop).resize((832,480),Image.Resampling.BICUBIC)
        path=OUT/'source_views'/f'{i:02d}.png';processed.save(path);source_paths.append(path)
    k_source=np.array(k_source);kt=np.array([[np.median(k_source[:,0,0]),0,415.5],[0,np.median(k_source[:,1,1]),239.5],[0,0,1.]])
    radius=float(np.median(np.linalg.norm(source_local[:,:2],axis=1)));hcam=float(np.median(source_local[:,2]))
    assert hcam>0,'Cameras should look slightly down at the body center'
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@axes.T+origin
    angles=[45,67.5,90,112.5,135]
    for _ in range(30):
        targets=[];positions=[];bounds=[]
        for angle in angles:
            theta=np.deg2rad(angle);p=center+radius*(np.cos(theta)*front-np.sin(theta)*left)+hcam*up
            w=look_at(p,center,up);uv,z=project(corners,w,kt)
            targets.append(w);positions.append(p);bounds.append([*uv.min(0),*uv.max(0)])
        if all(z>0 for z in [project(corners,w,kt)[1].min() for w in targets]) and all(b[0]>65 and b[2]<767 and b[1]>45 and b[3]<435 for b in bounds):break
        radius*=1.05
    else:raise ValueError('Target camera framing failed')
    trajectory=dict(source_w2c=m['source_w2c'],source_intrinsics=k_source.tolist(),
        target_w2c=np.asarray(targets).tolist(),target_intrinsics=[kt.tolist() for _ in angles],
        intrinsics_image_size=[832,480],camera_convention='OpenCV W2C')
    save(OUT/'trajectory.json',trajectory)
    car_errors=[p['groups']['car']['median_sampson_px'] for p in checks if p['groups']['car']['matches']>=10]
    background_errors=[p['groups']['background']['median_sampson_px'] for p in checks]
    audit=dict(ground_inlier_fraction=float(best.mean()),ground_tolerance=tolerance,center=center.tolist(),
        front=front.tolist(),left=left.tolist(),up=up.tolist(),ground_origin=origin.tolist(),
        observed_car_bounds_local=[lo.tolist(),hi.tolist()],observed_car_dimensions=[length,width_car,height],
        source_centers=centers.tolist(),source_coordinates_vehicle=source_local.tolist(),
        target_angles_deg=angles,target_positions=np.asarray(positions).tolist(),target_radius=radius,
        target_height_relative_to_body_center=hcam,target_bounds_model_pixels=bounds,source_crops=crops,
        median_car_sampson_px=float(np.median(car_errors)) if car_errors else None,
        median_background_sampson_px=float(np.median(background_errors)),
        all_targets_on_right=bool(np.all((np.asarray(positions)-center)@left<0)),
        limitations='Manual visible-car polygons and side-view head/tail regions; inferred center from observed partial point bounds, not hidden-side geometry GT; relative scale')
    save(OUT/'geometry_check.json',audit)
    # Independent SfM reference: nearby registered times, approximate comparison only.
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location('colmap_io_unseen',REPO/'thirdparty/map-anything/mapanything/utils/colmap.py')
    colmap=importlib.util.module_from_spec(spec);sys.modules[spec.name]=colmap;spec.loader.exec_module(colmap)
    root=REPO/'pipeline_exp/realworld/scene-1-nofull/3dgs_wcar'
    old=colmap.read_images_binary(root/'colmap/sparse/0/images.bin')
    old_meta=json.loads((root/'frames_manifest.json').read_text())
    byname={x.name:x for x in old.values()};old_centers=[];near=[]
    for item in m['sources']:
        nearest=min(old_meta['images'],key=lambda x:abs(x['video_frame_index']-item['source_frame_zero_based']))
        im=byname[nearest['image']];rotation=colmap.qvec2rotmat(im.qvec)
        old_centers.append(-rotation.T@im.tvec)
        near.append(dict(candidate=item['candidate'],sfm_image=im.name,delta_seconds=nearest['timestamp_sec']-item['pts_seconds']))
    from vggtdiff.omega import _similarity_transform
    scale,rot,trans=_similarity_transform(centers,np.array(old_centers))
    fitted=scale*(centers@rot.T)+trans
    norm=np.median(np.linalg.norm(np.array(old_centers)[:,None]-np.array(old_centers)[None],axis=-1)[np.triu_indices(6,1)])
    sfm_check=dict(registered_images=len(old),matching=near,
        aligned_center_rmse_relative_to_baseline=float(np.sqrt(np.mean(np.sum((fitted-np.array(old_centers))**2,axis=1)))/norm),
        note='Existing independent SfM of the same video; nearest registered frames are up to ~0.5s apart, not exact reference poses or GT')
    save(OUT/'sfm_crosscheck.json',sfm_check)
    fig,ax=plt.subplots(figsize=(10,8))
    cloud=(car-origin)@axes;ax.scatter(cloud[::10,0],cloud[::10,1],s=1,c='gray',alpha=.25,label='Observed car points')
    c=(center-origin)@axes;ax.plot(c[0],c[1],'kx',ms=10,label='Estimated body center')
    ax.arrow(c[0],c[1],length*.55,0,width=length*.007,color='green',length_includes_head=True)
    ax.text(c[0]+length*.6,c[1],'FRONT',color='green')
    sl=(centers-origin)@axes;tl=(np.array(positions)-origin)@axes
    ax.scatter(sl[:,0],sl[:,1],c='royalblue',marker='^',s=60,label='Real source cameras (left)')
    ax.scatter(tl[:,0],tl[:,1],c='crimson',marker='^',s=60,label='Requested cameras (right)')
    for i,p in enumerate(sl):
        ax.annotate(str(SELECTED[i]),p[:2],xytext=(6,5),textcoords='offset points',color='royalblue')
        ax.plot([p[0],c[0]],[p[1],c[1]],c='royalblue',lw=.5,alpha=.3)
    for angle,p in zip(angles,tl):
        ax.annotate(f'{angle:g} deg',p[:2],xytext=(5,-12),textcoords='offset points',color='crimson')
        ax.plot([p[0],c[0]],[p[1],c[1]],c='crimson',lw=.5,alpha=.3)
    ax.set_aspect('equal');ax.set_xlabel('Vehicle forward axis (relative units)');ax.set_ylabel('Vehicle LEFT axis (relative units)')
    ax.set_title('Real video source cameras and unseen RIGHT-side targets (top view)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'camera_layout.png',dpi=160);plt.close(fig)
    sheet(source_paths,[f'source candidate {i}' for i in SELECTED],OUT/'model_source_contact.jpg',3,400)
    save(OUT/'status.json',dict(state='geometry_checked_pending_generation',target_count=5,
        source_candidates=SELECTED,all_targets_on_right=audit['all_targets_on_right']))
    print(json.dumps(dict(geometry=audit,sfm=sfm_check),indent=2),flush=True)


def estimate():
    import torch
    from vggtdiff.omega import OmegaRuntime, _unproject
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio

    if (OUT/'camera_estimate.json').exists():raise FileExistsError('Camera estimate frozen')
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((INPUT/'candidates_manifest.json').read_text())
    selected=[meta['candidates'][i-1] for i in SELECTED]
    images=[]
    for item in selected:
        path=INPUT/item['full_image'];assert digest(path)==item['full_sha256']
        with Image.open(path) as im:images.append(im.convert('RGB'))
    save(OUT/'status.json',dict(state='estimating_omega',selected_candidates=SELECTED))
    start=time.time();omega=OmegaRuntime(str(OMEGA),torch.device('cuda'))
    tensors=omega._preprocess(images)
    with torch.inference_mode():
        result=omega.model(tensors)
        w,k=encoding_to_camera(result['pose_enc'],tensors.shape[-2:])
        world=_unproject(result['depth'][0].float(),w[0].float(),k[0].float())
    cameras=np.repeat(np.eye(4,dtype=np.float32)[None],6,axis=0)
    cameras[:,:3]=w[0].cpu().numpy();ks=k[0].cpu().numpy();original_k=[];maps=[]
    h,width=tensors.shape[-2:]
    for image,intrinsic in zip(images,ks):
        cw,ch=_crop_to_supported_aspect_ratio(image).size
        left=(image.width-cw)//2;top=(image.height-ch)//2
        a=np.array([[cw/width,0,left],[0,ch/h,top],[0,0,1.]])
        original_k.append((a@intrinsic).tolist());maps.append(a.tolist())
    xyz=world.cpu().numpy();conf=result['depth_conf'][0].cpu().numpy()
    if conf.ndim==4:conf=conf[...,0]
    rgb=(tensors[0].permute(0,2,3,1).float().cpu().numpy()*255).round().clip(0,255).astype('uint8')
    np.savez_compressed(OUT/'omega_dense.npz',world=xyz,confidence=conf,
        depth=result['depth'][0].cpu().numpy(),rgb=rgb,w2c=cameras,k_processed=ks,
        k_original=np.asarray(original_k),processed_to_original=np.asarray(maps))
    centers=np.linalg.inv(cameras)[:,:3,3]
    baseline=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
    assert np.isfinite(cameras).all() and np.isfinite(original_k).all()
    assert np.max(np.abs(np.linalg.det(cameras[:,:3,:3])-1))<1e-3
    assert np.median(baseline[np.triu_indices(6,1)])>1e-5
    save(OUT/'camera_estimate.json',dict(method='VGGT-Omega',selected_candidates=SELECTED,
        sources=selected,source_root=str(INPUT),source_w2c=cameras.tolist(),
        source_intrinsics=original_k,original_resolution=[3840,2160],
        processed_resolution=[width,h],processed_to_original=maps,
        elapsed_seconds=time.time()-start,coordinate_convention='OpenCV W2C',
        scale='arbitrary common scale; no metric calibration',median_baseline=float(np.median(baseline[np.triu_indices(6,1)]))))
    sheet([INPUT/x['crop_image'] for x in selected],[f"candidate {x['candidate']:02d} | {x['pts_seconds']:.2f}s" for x in selected],OUT/'selected_sources_preview.jpg',3,400)
    save(OUT/'status.json',dict(state='estimated_pending_geometry_check',selected_candidates=SELECTED))
    print('ESTIMATED',time.time()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['estimate','calibrate','check_crop','rectify','infer'])
    args=parser.parse_args()
    {'estimate':estimate,'calibrate':calibrate,'check_crop':check_crop,'rectify':rectify,'infer':infer}[args.action]()
