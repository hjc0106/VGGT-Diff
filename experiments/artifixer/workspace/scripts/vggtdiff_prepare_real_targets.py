"""Localize held-out real video frames and export camera-matched NVS targets.

CPU only. Background PnP uses the existing real-only Omega world, not camera GT.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

import vggtdiff_realvideo_unseen as e
from vggtdiff_white_car import BASE, REPO, digest, save

OUT = BASE / 'vggtdiff_real_gt_batch06_targets_v1'
REAL = BASE / 'vggtdiff_realvideo_unseen_right_near_v2'
DENSE = BASE / 'vggtdiff_realvideo_unseen_right_v1/omega_dense.npz'
VIDEO = REPO / 'pipeline_exp/realworld/scene-1-nofull/scene-1-wcar.mp4'
CANDIDATES = [910, 940, 980, 1020, 1060, 1100, 1140, 1180, 1220, 1260, 1300, 1340]


def read(p):
    return json.loads(p.read_text())


def azimuth(w, geometry):
    d = np.linalg.inv(w)[:3, 3] - np.array(geometry['center'])
    return float(np.degrees(np.arctan2(-d @ np.array(geometry['left']),
                                     d @ np.array(geometry['front']))) % 360)


def background_mask(shape, polygon):
    car = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(car, [np.rint(polygon).astype(np.int32)], 255)
    return 255 - cv2.dilate(car, np.ones((41, 41), np.uint8))


def make_references(meta, sift):
    frames = np.array([r['source_frame_zero_based'] for r in meta['sources']])
    polygons = np.asarray(e.POLYGONS, dtype=object)
    # Source polygons have different vertex counts: interpolate bounding boxes only.
    boxes = np.array([[np.min(p, axis=0), np.max(p, axis=0)] for p in polygons], dtype=float) * 4 / 3
    cache = OUT / 'reference_features_sift32k_conf40_v2.npz'
    if cache.exists():
        with np.load(cache) as data:
            return [(data[f'xyz{i}'], data[f'des{i}']) for i in range(6)], frames, boxes
    references = []
    values = {}
    with np.load(DENSE) as dense:
        for i, item in enumerate(meta['sources']):
            path = e.INPUT / item['full_image']
            if digest(path) != item['full_sha256']:
                raise ValueError(f'Source changed: {path}')
            image = cv2.resize(cv2.imread(str(path)), (1280, 720))
            mask = background_mask(image.shape, np.asarray(e.POLYGONS[i]) * 4 / 3)
            kp, des = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
            xy = np.array([p.pt for p in kp])
            original = np.c_[xy * 3, np.ones(len(xy))]
            processed = original @ np.linalg.inv(dense['processed_to_original'][i]).T
            uv = np.rint(processed[:, :2] / processed[:, 2:]).astype(int)
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < 688) & (uv[:, 1] >= 0) & (uv[:, 1] < 384)
            ids = np.flatnonzero(valid)
            uv = uv[ids]
            conf = dense['confidence'][i]
            keep = conf[uv[:, 1], uv[:, 0]] >= np.quantile(conf, .4)
            ids, uv = ids[keep], uv[keep]
            xyz = dense['world'][i, uv[:, 1], uv[:, 0]]
            finite = np.isfinite(xyz).all(1)
            xyz, des = xyz[finite].copy(), des[ids[finite]].copy()
            references.append((xyz, des))
            values[f'xyz{i}'], values[f'des{i}'] = xyz, des
            print('REFERENCE', i, len(xyz), flush=True)
    np.savez_compressed(cache, **values)
    return references, frames, boxes


def decode():
    full = OUT / 'full'
    full.mkdir(parents=True, exist_ok=True)
    pending = [f for f in CANDIDATES if not (full / f'{f}.png').exists()]
    if not pending:
        return
    selected = '+'.join(f'eq(n,{f})' for f in pending)
    temp = OUT / 'decoded'
    temp.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(VIDEO),
                    '-vf', f"select='{selected}'", '-vsync', '0', '-threads', '2',
                    '-start_number', '0', '-frames:v', str(len(pending)), '-y', str(temp / '%03d.png')], check=True)
    paths = sorted(temp.glob('*.png'))
    if len(paths) != len(pending):
        raise ValueError('Decoded frame count mismatch')
    for path, frame in zip(paths, pending):
        path.rename(full / f'{frame}.png')


def localize(frame, references, frames, boxes, kfull, geometry, virtual_k, sift):
    cv2.setRNGSeed(20261022 + frame)
    original = cv2.imread(str(OUT / 'full' / f'{frame}.png'))
    small = cv2.resize(original, (1280, 720))
    box = np.array([np.interp(frame, frames, boxes[:, corner, axis])
                    for corner in range(2) for axis in range(2)]).reshape(2, 2)
    (x0, y0), (x1, y1) = box
    polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    mask = background_mask(small.shape, polygon)
    kp, des = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), mask)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=128))
    matcher.add([des])
    matcher.train()
    best = {}
    for xyz, reference_des in references:
        matches = matcher.knnMatch(reference_des, k=2)
        for pair in matches:
            if len(pair) != 2:
                continue
            a, b = pair
            if a.distance < .75 * b.distance and (a.trainIdx not in best or a.distance < best[a.trainIdx][0]):
                best[a.trainIdx] = (a.distance, xyz[a.queryIdx])
    uv = np.asarray([kp[idx].pt for idx in best], dtype=np.float32)
    xyz = np.asarray([value[1] for value in best.values()], dtype=np.float32)
    # Reserve target spatial cells before estimating pose; these matches never fit PnP.
    heldout = ((uv[:, 0] // 80 + 7 * (uv[:, 1] // 80)).astype(int) % 5) == 0
    train_ids = np.flatnonzero(~heldout)
    ksmall = kfull.copy()
    ksmall[:2] /= 3
    ok, rv, tv, inliers = cv2.solvePnPRansac(xyz[train_ids], uv[train_ids], ksmall, None,
        iterationsCount=10000, reprojectionError=2.5, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or len(inliers) < 20:
        return dict(original_frame=frame, state='failed_pnp', matches=len(xyz))
    ids = train_ids[inliers[:, 0]]
    rv, tv = cv2.solvePnPRefineLM(xyz[ids], uv[ids], ksmall, None, rv, tv)
    projected = cv2.projectPoints(xyz, rv, tv, ksmall, None)[0][:, 0]
    errors = np.linalg.norm(projected - uv, axis=1)
    test_errors = errors[heldout]
    heldout_good = int(np.sum(test_errors < 3))
    w = np.eye(4)
    w[:3, :3], w[:3, 3] = cv2.Rodrigues(rv)[0], tv[:, 0]
    position = np.linalg.inv(w)[:3, 3]
    neww = e.look_at(position, np.array(geometry['center']), np.array(geometry['up']))
    h = virtual_k @ neww[:3, :3] @ w[:3, :3].T @ np.linalg.inv(kfull)
    warped = cv2.warpPerspective(original, h, (832, 480), flags=cv2.INTER_CUBIC)
    validity = cv2.warpPerspective(np.ones(original.shape[:2], np.uint8), h, (832, 480), flags=cv2.INTER_NEAREST)
    path = OUT / 'candidate_images' / f'{frame}.png'
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), warped)
    # Numerically check that rotational resampling and exported cameras encode identical rays.
    test_xyz = xyz[ids[:50]].astype(float)
    before, _ = e.project(test_xyz, w, kfull)
    after, _ = e.project(test_xyz, neww, virtual_k)
    mapped = np.c_[before, np.ones(len(before))] @ h.T
    ray_error = float(np.max(np.abs(mapped[:, :2] / mapped[:, 2:] - after)))
    span = np.ptp(uv[ids], axis=0)
    good = (heldout_good >= 8 and np.median(errors[ids]) < 1.5
            and np.quantile(errors[ids], .95) < 3 and validity.mean() > .999
            and min(span) > 100 and ray_error < 1e-7)
    np.savez_compressed(OUT / 'matches' / f'{frame}.npz', xyz=xyz, uv=uv,
                        heldout=heldout, pnp_inlier_ids=ids, errors=errors)
    return dict(original_frame=frame, state='localized' if good else 'low_support',
        angle=azimuth(neww, geometry), matches=len(xyz), inliers=len(ids),
        heldout_matches=int(heldout.sum()), heldout_under_3px=heldout_good,
        heldout_raw_median_px=float(np.median(test_errors)) if len(test_errors) else None,
        median_reprojection_error_1280_px=float(np.median(errors[ids])),
        p95_reprojection_error_1280_px=float(np.quantile(errors[ids], .95)),
        inlier_span_1280_px=span.tolist(), valid_fraction=float(validity.mean()),
        homography_ray_check_max=ray_error, path=str(path), sha256=digest(path),
        full_path=str(OUT / 'full' / f'{frame}.png'), full_sha256=digest(OUT / 'full' / f'{frame}.png'),
        w2c=neww.tolist(), k=virtual_k.tolist(), full_w2c=w.tolist(), full_k=kfull.tolist(),
        homography=h.tolist())


def main():
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'matches').mkdir(exist_ok=True)
    meta, geometry = read(REAL / 'camera_estimate.json'), read(REAL / 'geometry_check.json')
    source_t = read(REAL / 'trajectory.json')
    source_frames = [r['source_frame_zero_based'] for r in meta['sources']]
    assert not set(CANDIDATES) & set(source_frames)
    with np.load(DENSE) as dense:
        kfull = np.median(dense['k_original'], axis=0)
    # Match GT framing to the real conditioning images; all virtual pixels are resampled real pixels.
    virtual_k = np.median(np.array(source_t['source_intrinsics']), axis=0)
    virtual_k[0, 2], virtual_k[1, 2] = 415.5, 239.5
    sift = cv2.SIFT_create(nfeatures=32000, contrastThreshold=.02)
    references, frames, boxes = make_references(meta, sift)
    decode()
    rows = []
    for frame in CANDIDATES:
        dest = OUT / f'localization_{frame}.json'
        row = read(dest) if dest.exists() else localize(frame, references, frames, boxes, kfull, geometry, virtual_k, sift)
        save(dest, row)
        rows.append(row)
        print('LOCALIZED', frame, row['state'], row.get('angle'), row.get('inliers'), row.get('heldout_under_3px'), flush=True)
    save(OUT / 'candidates.json', dict(frames=rows, source_frames=source_frames,
        calibration_caution='Background-only PnP in existing estimated real-only Omega world; not calibrated pose ground truth. Heldout spatial cells never fit PnP. Intrinsics assumed from existing estimates; lens distortion not independently calibrated.'))
    finalize(rows, source_t, geometry, source_frames)


def finalize(rows, source_t, geometry, source_frames):
    import shutil
    from scipy.optimize import linear_sum_assignment
    from vggtdiff_latest_coverage_ab import montage

    eligible = [r for r in rows if r['state'] == 'localized' and 223 <= r['angle'] <= 267]
    if len(eligible) < 8:
        raise ValueError(f'Only {len(eligible)} valid target candidates; need eight')
    desired = np.array([225 + 5.625 * i for i in range(8)])
    cost = np.abs(desired[:, None] - np.array([r['angle'] for r in eligible])[None])
    a, b = linear_sum_assignment(cost)
    selected = [dict(eligible[j]) for i, j in sorted(zip(a, b))]
    # Keep a shared target K, with enough margin for the full real vehicle.
    # Existing conditioning framing crops the rear bumper in some held-out views.
    meta = read(REAL / 'camera_estimate.json')
    source_numbers = np.array([r['source_frame_zero_based'] for r in meta['sources']])
    boxes = np.array([[np.min(p, axis=0), np.max(p, axis=0)] for p in e.POLYGONS], dtype=float) * 4
    target_boxes = []
    for r in selected:
        box = np.array([np.interp(r['original_frame'], source_numbers, boxes[:, c, axis])
                        for c in range(2) for axis in range(2)]).reshape(2, 2)
        pad = .06 * (box[1] - box[0])
        (x0, y0), (x1, y1) = box + np.array([-pad, pad])
        target_boxes.append(np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]))
    common_k = np.array(selected[0]['k'])
    for _ in range(60):
        bounds = []
        for r, corners in zip(selected, target_boxes):
            w, full_w, full_k = np.array(r['w2c']), np.array(r['full_w2c']), np.array(r['full_k'])
            h = common_k @ w[:3, :3] @ full_w[:3, :3].T @ np.linalg.inv(full_k)
            uv = corners @ h.T
            uv = uv[:, :2] / uv[:, 2:]
            bounds.append(np.r_[uv.min(0), uv.max(0)])
        if all(b[0] >= 20 and b[1] >= 20 and b[2] <= 812 and b[3] <= 460 for b in bounds):
            break
        common_k[0, 0] *= .97
        common_k[1, 1] *= .97
    else:
        raise ValueError('Could not frame all target vehicles')
    images = OUT / 'gt'
    images.mkdir(exist_ok=True)
    for i, r in enumerate(selected):
        destination = images / f'{i:02d}.png'
        w, full_w, full_k = np.array(r['w2c']), np.array(r['full_w2c']), np.array(r['full_k'])
        h = common_k @ w[:3, :3] @ full_w[:3, :3].T @ np.linalg.inv(full_k)
        original = cv2.imread(r['full_path'])
        warped = cv2.warpPerspective(original, h, (832, 480), flags=cv2.INTER_CUBIC)
        validity = cv2.warpPerspective(np.ones(original.shape[:2], np.uint8), h, (832, 480), flags=cv2.INTER_NEAREST)
        assert validity.mean() > .999
        if destination.exists():
            old = cv2.imread(str(destination))
            if not np.array_equal(old, warped):
                raise ValueError(f'Existing target differs: {destination}')
        cv2.imwrite(str(destination), warped)
        r['k'], r['homography'] = common_k.tolist(), h.tolist()
        r['path'], r['sha256'] = str(destination), digest(destination)
        r['valid_fraction'] = float(validity.mean())
        r['conservative_vehicle_bounds_virtual'] = bounds[i].tolist()
        r['target_index'] = i
        r['selected_gt_path'] = str(destination)
        r['original_batch6_angle'] = float(desired[i])
        r['difference_to_original_batch6_degrees'] = float(r['angle'] - desired[i])
        # A rotation-only image warp leaves camera position and azimuth unchanged.
        assert np.allclose(np.linalg.inv(r['w2c'])[:3, 3], np.linalg.inv(r['full_w2c'])[:3, 3])
        assert r['original_frame'] not in source_frames
    source_dir = OUT / 'source_views'
    source_dir.mkdir(exist_ok=True)
    for i in range(6):
        src, dst = REAL / 'source_views' / f'{i:02d}.png', source_dir / f'{i:02d}.png'
        if dst.exists() and digest(src) != digest(dst):
            raise ValueError(f'Existing reference differs: {dst}')
        shutil.copyfile(src, dst)
    trajectory = dict(source_w2c=source_t['source_w2c'], source_intrinsics=source_t['source_intrinsics'],
        target_w2c=[r['w2c'] for r in selected], target_intrinsics=[r['k'] for r in selected],
        target_angles_deg=[r['angle'] for r in selected],
        target_video_frames=[r['original_frame'] for r in selected],
        intrinsics_image_size=[832, 480], camera_convention='OpenCV W2C',
        note='Target cameras belong to held-out real frames, not the original uniform orbit. Reuse exactly these cameras for matched generation.')
    save(OUT / 'trajectory.json', trajectory)
    save(OUT / 'cameras.json', dict(frames=selected, camera_status='estimated_not_calibrated_ground_truth',
        image_status='Real decoded video pixels, resampled by calibrated rotational homography; no generative RGB.',
        background_pnp=True, vehicle_feature_exclusion='Conservative interpolated vehicle bounding boxes, expanded by 20 pixels at 1280x720; source vehicle polygons expanded similarly.',
        holdout_protocol='Spatial cells excluded before PnP, not an independent sensor or reference-map truth.',
        distortion_caution='Existing estimated pinhole intrinsics; original video distortion was not independently calibrated.'))
    montage([Path(r['selected_gt_path']) for r in selected],
            [f"GT {i:02d} | frame {r['original_frame']} | {r['angle']:.2f} deg" for i, r in enumerate(selected)],
            OUT / 'gt_contact8.jpg', columns=4)
    file_list = [REAL / 'trajectory.json', REAL / 'camera_estimate.json', REAL / 'geometry_check.json',
                 DENSE, Path(__file__).resolve(), OUT / 'candidates.json', OUT / 'cameras.json', OUT / 'trajectory.json']
    file_list += list(source_dir.glob('*.png')) + list(images.glob('*.png'))
    file_list += [Path(r['full_path']) for r in selected]
    file_list += [e.INPUT / r['full_image'] for r in meta['sources']]
    file_list += [OUT / 'reference_features_sift32k_conf40_v2.npz']
    file_list += [OUT / 'matches' / f"{r['original_frame']}.npz" for r in selected]
    manifest = dict(stage='prepared_pending_visual_review', selected_frames=[r['original_frame'] for r in selected],
        source_frames=source_frames, angle_range=[min(trajectory['target_angles_deg']), max(trajectory['target_angles_deg'])],
        video=dict(path=str(VIDEO), size=VIDEO.stat().st_size, mtime_ns=VIDEO.stat().st_mtime_ns),
        frozen_files=[dict(path=str(p), sha256=digest(p)) for p in file_list])
    save(OUT / 'manifest.json', manifest)
    report = ['# 第六批范围的真实留出目标', '',
        '已提取八张未用作六张条件参考的真实视频帧，并通过背景特征定位到原有真实 Omega 世界。相机是估计值，不是标定真值。', '',
        '| 目标 | 原视频帧 | 估计方位角 | 拟合内点 | 留出背景匹配 <3px | 拟合误差 median/P95（1280×720） |',
        '|---|---:|---:|---:|---:|---:|']
    for i, r in enumerate(selected):
        report.append(f"| {i:02d} | {r['original_frame']} | {r['angle']:.3f}° | {r['inliers']} | {r['heldout_under_3px']}/{r['heldout_matches']} | {r['median_reprojection_error_1280_px']:.3f}/{r['p95_reprojection_error_1280_px']:.3f} px |")
    report += ['', 'GT 文件为 gt/00.png–07.png；同视角生成必须使用本目录 trajectory.json 中的全部 source/target 相机。source_views 为原六张真实条件图。',
        '', '这些目标改变了原均匀环视的相机中心、方位角和内参；正在运行的225°–264.375°环视结果不能与这些GT当作同相机逐像素配对。按角度近邻只能作近似视觉参考。',
        '', '旋转单应性仅改变朝向与内参，保持真实相机中心；不把真实图像伪装成任意环视位置。所有虚拟像素均来自原图，valid_fraction>0.999，并进行了投影/射线数值一致性检查。',
        '', '限制：背景只使用膨胀车辆排除区，其他运动物体由RANSAC抵抗而非人工逐个分割；VGGT深度、固定内参、镜头畸变与平面退化均可能影响定位。留出背景匹配检查不能消除参考几何的系统误差。未开始生成、分割或几何指标评估。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(report))
    save(OUT / 'status.json', dict(state='prepared_pending_visual_review', frames=8))
    print('PREPARED_TARGETS', json.dumps(trajectory['target_video_frames']), json.dumps(trajectory['target_angles_deg']), flush=True)


if __name__ == '__main__':
    main()
