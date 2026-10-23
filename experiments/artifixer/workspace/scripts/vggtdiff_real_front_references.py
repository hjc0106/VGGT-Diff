"""Rectify registered real front-view cameras into the existing Omega world.

True original video pixels only. Correct SfM OPENCV distortion and reorient at
unchanged camera centers. Zoom the virtual pinhole only to avoid invalid rays.
"""
from pathlib import Path
import cv2
import numpy as np
import vggtdiff_real_front_sfm_bridge as bridge
from vggtdiff_white_car import BASE,REPO,save,digest

OUT=bridge.anchors.OUT


def prepare_real(output=OUT):
    cv2.setNumThreads(4)
    calibration=output/'real_calibration';align_path=bridge.OUT/'alignment.json'
    if not align_path.exists():bridge.fit_bridge()
    alignment=bridge.anchors.gt.read(align_path)
    if alignment['heldout_matches']<50 or alignment['heldout_median_1280_px']>3 or alignment['heldout_p90_1280_px']>8:
        raise ValueError('SfM/Omega bridge lacks reliable heldout support')
    scale=alignment['scale_sfm_to_omega'];rotate=np.array(alignment['rotation_sfm_to_omega']);shift=np.array(alignment['translation_sfm_to_omega'])
    c=bridge.load_colmap();images=c.read_images_binary(bridge.ROOT/'colmap/sparse/0/images.bin');byname={r.name:r for r in images.values()}
    cameras=c.read_cameras_binary(bridge.ROOT/'colmap/sparse/0/cameras.bin');points=c.read_points3d_binary(bridge.ROOT/'colmap/sparse/0/points3D.bin')
    metadata=bridge.anchors.gt.read(bridge.ROOT/'frames_manifest.json')['images'];byframe={r['video_frame_index']:r for r in metadata}
    geometry=bridge.anchors.gt.read(bridge.anchors.gt.REAL/'geometry_check.json');center=np.array(geometry['center']);up=np.array(geometry['up'])
    old=bridge.anchors.gt.read(BASE/'vggtdiff_batch06_07gen_to_batch08_continuous_v1/trajectory.json');base_k=np.array(old['target_intrinsics'][0])
    rows=[]
    for frame in bridge.anchors.FRAMES:
        record_path=calibration/f'camera_{frame}.json'
        if record_path.exists():
            row=bridge.anchors.gt.read(record_path)
            if row['alignment_sha256']!=digest(align_path):raise ValueError('Real camera cache uses a different alignment')
            if digest(Path(row['image']))!=row['sha256']:raise ValueError('Real reference changed')
            rows.append(row);continue
        meta=byframe[frame];im=byname[meta['image']];camera=cameras[im.camera_id]
        if camera.model!='OPENCV':raise ValueError('Expected OPENCV source calibration')
        rs=c.qvec2rotmat(im.qvec);ts=im.tvec
        full_w=np.eye(4);full_w[:3,:3]=rs@rotate.T;full_w[:3,3]=scale*ts-full_w[:3,:3]@shift
        pos=np.linalg.inv(full_w)[:3,3];virtual_w=bridge.anchors.gt.e.look_at(pos,center,up)
        params=camera.params;kfull=np.array([[params[0],0,params[2]],[0,params[1],params[3]],[0,0,1.]])
        distortion=params[4:];relative=virtual_w[:3,:3]@full_w[:3,:3].T
        raw=calibration/'full'/f'{frame}.png'
        if not raw.exists():
            raw.parent.mkdir(parents=True,exist_ok=True);capture=cv2.VideoCapture(str(bridge.anchors.VIDEO));capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,rgb=capture.read();capture.release()
            if not ok:raise ValueError(f'Cannot read real frame{frame}')
            cv2.imwrite(str(raw),rgb)
        rgb=cv2.imread(str(raw));kn=base_k.copy()
        for _ in range(70):
            mx,my=cv2.initUndistortRectifyMap(kfull,distortion,relative,kn,(832,480),cv2.CV_32FC1)
            valid=(mx>=2)&(mx<3837)&(my>=2)&(my<2157)
            if valid.mean()>.99999:break
            kn[0,0]*=1.025;kn[1,1]*=1.025
        else:raise ValueError('Could not obtain fully supported reference rays')
        warped=cv2.remap(rgb,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        destination=output/'real_views'/f'{frame}.png';destination.parent.mkdir(exist_ok=True);cv2.imwrite(str(destination),warped)
        ids=im.point3D_ids>=0;observed=im.xys[ids];xyz=np.array([points[int(i)].xyz for i in im.point3D_ids[ids]])
        uv=cv2.projectPoints(xyz,cv2.Rodrigues(rs)[0],ts,kfull,distortion)[0][:,0]
        sfm_error=np.linalg.norm(uv-observed,axis=1)
        # The actual distortion correction, virtual rotation and exported cameras must agree.
        transformed=scale*(xyz@rotate.T)+shift
        projected=bridge.project(transformed,virtual_w,kn)
        ray_uv=cv2.undistortPoints(uv[:,None],kfull,distortion,R=relative,P=kn)[:,0]
        ray_error=float(np.abs(projected-ray_uv).max())
        assert ray_error<1e-3,ray_error
        np.testing.assert_allclose(np.linalg.inv(virtual_w)[:3,3],pos,atol=1e-12)
        angle=bridge.anchors.gt.azimuth(virtual_w,geometry)
        row=dict(original_frame=frame,role='real',angle=angle,image=str(destination),sha256=digest(destination),
            full_image=str(raw),full_sha256=digest(raw),w2c=virtual_w.tolist(),intrinsics=kn.tolist(),
            original_w2c=full_w.tolist(),original_intrinsics=kfull.tolist(),distortion=distortion.tolist(),
            virtual_rotation=relative.tolist(),valid_fraction=float(valid.mean()),alignment_sha256=digest(align_path),
            sfm_observations=len(sfm_error),sfm_reprojection_native_median_px=float(np.median(sfm_error)),sfm_reprojection_native_p90_px=float(np.percentile(sfm_error,90)),
            distortion_rotation_projection_check_max_px=ray_error,camera_status='SfM registered camera mapped by a background-pixel-supported Sim3; estimated, not calibrated GT')
        save(record_path,row);rows.append(row)
        print('REAL_REFERENCE',frame,'angle',angle,'fx',kn[0,0],'valid',valid.mean(),'sfm_matches',len(sfm_error),'median',np.median(sfm_error),'ray_check',ray_error,flush=True)
    save(calibration/'real_references.json',dict(frames=rows,alignment=alignment,
        processing='Original real video pixels, distortion correction plus rotation about unchanged real centers. Intrinsics updated for output832x480. No synthesized pixels.'))
    return rows

if __name__=='__main__':prepare_real()
