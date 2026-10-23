"""Background-only real-video camera localization for batch8 reference anchors.

Reuse the frozen real6 Omega map, SIFT32k, spatially-held-out PnP protocol.
Source images are rotation-only resamples at unchanged real camera centers.
"""
from pathlib import Path
import cv2
import numpy as np
import vggtdiff_prepare_real_targets as gt
from vggtdiff_white_car import BASE,REPO,save,digest

OUT=BASE/'vggtdiff_batch08_real4_gen2_continuous_v1'
FRAMES=[2872,2751,2691,2630]
COVERAGE=BASE/'realvideo_batch08_coverage_v1/real_front_candidates.json'
CACHE=BASE/'vggtdiff_real_gt_batch06_targets_v1/reference_features_sift32k_conf40_v2.npz'
VIDEO=REPO/'pipeline_exp/realworld/scene-1-nofull/scene-1-wcar.mp4'


def calibrate(output=OUT):
    cv2.setNumThreads(4)
    root=output/'real_calibration';(root/'full').mkdir(parents=True,exist_ok=True);(root/'matches').mkdir(exist_ok=True)
    candidates={r['video_frame_index']:r for r in gt.read(COVERAGE)['selected_frames']}
    geometry=gt.read(gt.REAL/'geometry_check.json')
    with np.load(CACHE) as data:references=[(data[f'xyz{i}'],data[f'des{i}']) for i in range(6)]
    with np.load(gt.DENSE) as data:kfull=np.median(data['k_original'],axis=0)
    baseline=gt.read(BASE/'vggtdiff_batch06_07gen_to_batch08_continuous_v1/trajectory.json')
    base_k=np.asarray(baseline['target_intrinsics'][0])
    sift=cv2.SIFT_create(nfeatures=32000,contrastThreshold=.02)
    gt.OUT=root
    rows=[];cap=None
    for f in FRAMES:
        full=root/'full'/f'{f}.png'
        if not full.exists():
            if cap is None:cap=cv2.VideoCapture(str(VIDEO))
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,rgb=cap.read()
            if not ok:raise ValueError(f'Cannot decode real frame{f}')
            cv2.imwrite(str(full),rgb)
        path=root/f'localization_{f}.json'
        if path.exists():r=gt.read(path)
        else:
            # Exclude a deliberately broad vehicle ROI from target background matching.
            box=np.asarray(candidates[f]['crop_xyxy'],float).reshape(2,2)/3
            r=gt.localize(f,references,np.array([f]),box[None],kfull,geometry,base_k.copy(),sift)
            save(path,r)
        print('BACKGROUND_PNP',f,r.get('state'),r.get('angle'),r.get('inliers'),r.get('heldout_under_3px'),r.get('median_reprojection_error_1280_px'),r.get('p95_reprojection_error_1280_px'),r.get('valid_fraction'),flush=True)
        rows.append(r)
    if cap is not None:cap.release()
    save(root/'localizations.json',dict(frames=rows,method='Existing backgroundPnP in real-only Omega map; target vehicle ROI excluded; spatial holdout never used in fitting.',
        image_size_full=[3840,2160],camera_status='Estimated cameras; not independent calibrated GT.'))
    return rows

if __name__=='__main__':calibrate()
