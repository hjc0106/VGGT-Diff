"""Evaluate the eight dense held-out real targets, preserving raw diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

import crossview_geometry_eval as geom
from vggtdiff_white_car import BASE, REPO, digest, save
from vggtdiff_latest_coverage_ab import montage

SOURCE = BASE / 'vggtdiff_real_gt_batch06_targets_v1'
OUT = BASE / 'crossview_geometry_dense_real8_v1'


def read(path):
    return json.loads(path.read_text())


def validate():
    for row in read(OUT / 'manifest.json')['frozen_files']:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f"Frozen input changed: {row['path']}")


def prepare():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    from vggtdiff_car15_groupa_prepare import choose_mask

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'masks').mkdir(exist_ok=True)
    for row in read(SOURCE / 'manifest.json')['frozen_files']:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f"GT source changed: {row['path']}")
    rows = read(SOURCE / 'cameras.json')['frames']
    trajectory = read(SOURCE / 'trajectory.json')
    assert [r['original_frame'] for r in rows] == trajectory['target_video_frames']
    assert len(rows) == 8
    segmenter = SAM3Segmenter(SimpleNamespace(source_dir=str(REPO / 'thirdparty/sam3'),
        checkpoint_path=str(REPO / 'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
        device='cuda', confidence_threshold=.15, instance_strategy='best_score'))
    frames, overlays = [], []
    for i, row in enumerate(rows):
        image = Path(row['selected_gt_path'])
        if digest(image) != row['sha256']:
            raise ValueError(f'GT image changed: {image}')
        rgb = np.asarray(Image.open(image).convert('RGB'))
        mask_path = OUT / 'masks' / f'{i:02d}.png'
        if not mask_path.exists():
            mask = choose_mask(segmenter, rgb)
            Image.fromarray(mask.astype('uint8') * 255).save(mask_path)
        mask = np.asarray(Image.open(mask_path).convert('L')) > 127
        ys, xs = np.where(mask)
        assert mask.shape == (480, 832) and len(xs) > 1000
        overlay = rgb.copy()
        overlay[mask] = (.55 * rgb[mask] + .45 * np.array([0, 255, 0])).astype('uint8')
        cv2.drawContours(overlay, cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 0, 0), 1)
        overlay_path = OUT / 'masks' / f'{i:02d}_overlay.jpg'
        Image.fromarray(overlay).save(overlay_path)
        overlays.append(overlay_path)
        frames.append(dict(index=i, original_frame=row['original_frame'], image=str(image),
            mask=str(mask_path), sha256=digest(image), mask_sha256=digest(mask_path),
            K=row['k'], T=row['w2c'], azimuth_deg=row['angle'],
            mask_bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
            mask_area_fraction=float(mask.mean())))
        print('MASK', i, row['original_frame'], len(xs), flush=True)
    montage(overlays, [f"{r['index']:02d} | {r['original_frame']} | {r['azimuth_deg']:.2f} deg" for r in frames],
            OUT / 'mask_review.jpg', columns=4)
    pairs = [dict(frame_a=a['original_frame'], frame_b=b['original_frame'],
        index_a=a['index'], index_b=b['index'], index_gap=b['index']-a['index'],
        azimuth_gap_deg=geom.angular_gap(a['azimuth_deg'], b['azimuth_deg']))
        for a, b in itertools.combinations(frames, 2)]
    protocol = dict(dataset='dense held-out real8', image_size=[832, 480],
        camera_status='background PnP in estimated real-only Omega world; not pose GT',
        matcher=dict(type='SIFT', nfeatures=12000, ratio=.75, mutual=True, epipolar_prefilter=False),
        support=dict(minimum_matches=30, coverage_grid=[8, 6], status='exploratory reporting rule'),
        primary_groups=['adjacent seven pairs', '0-10 deg', '10-15 deg', '15-30 deg', '30+ deg'],
        track_groups=['global raw graph', 'six consecutive local triples raw graph',
                      'same local triples with direct three-edge descriptor cycle'],
        cycle_rule='All three mutual descriptor matches must directly agree. No reprojection-error selection.',
        parallax_deg=[.5, 1, 2], primary_parallax_deg=1,
        report_thresholds_px=[2, 4], thresholds_are_quality_acceptance_lines=False,
        failure_policy='Separate insufficient support, computational failures and reprojection-error distribution.',
        statistics='Pair medians equally weighted. Track attempts are pooled, overlapping triples are not independent samples.',
        scope='Real dense GT only; no generated-image metrics or new diffusion inference.')
    save(OUT / 'frames.json', dict(frames=frames))
    save(OUT / 'pairs_spec.json', dict(pairs=pairs))
    save(OUT / 'protocol.json', protocol)
    files = [SOURCE / 'manifest.json', SOURCE / 'cameras.json', SOURCE / 'trajectory.json',
             Path(__file__).resolve(), Path(geom.__file__).resolve(),
             OUT / 'frames.json', OUT / 'pairs_spec.json', OUT / 'protocol.json']
    files += [Path(r[key]) for r in frames for key in ['image', 'mask']]
    for name in ['crossview_geometry_eval.py', 'crossview_geometry_dense_gt.py']:
        dest = OUT / 'implementation' / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(REPO / 'scripts' / name, dest)
    manifest = dict(frames=[r['original_frame'] for r in frames],
        frozen_files=[dict(path=str(p), sha256=digest(p)) for p in files])
    if (OUT / 'manifest.json').exists() and read(OUT / 'manifest.json') != manifest:
        raise ValueError('Existing frozen experiment differs')
    save(OUT / 'manifest.json', manifest)
    save(OUT / 'status.json', dict(state='prepared_pending_mask_review'))


def pair_summary(rows):
    supported = [r for r in rows if not r['low_support'] and r['epi_median_px'] is not None]
    def median(key, subset):
        v = [r[key] for r in subset if r.get(key) is not None]
        return float(np.median(v)) if v else None
    return dict(n_pairs=len(rows), supported_pairs=len(supported), low_support_pairs=sum(r['low_support'] for r in rows),
        median_pair_epi_px=median('epi_median_px', rows),
        supported_median_pair_epi_px=median('epi_median_px', supported),
        supported_median_pair_p90_px=median('epi_p90_px', supported),
        median_matches=median('n_matches', rows),
        median_coverage_a=median('coverage_a', rows), median_coverage_b=median('coverage_b', rows))


def direct_cycle(track, edge_sets):
    if track['conflict'] or track['n_views'] != 3:
        return False
    for a, b in itertools.combinations(sorted(track['observations']), 2):
        if (track['observations'][a], track['observations'][b]) not in edge_sets.get((a, b), set()):
            return False
    return True


def summarized_attempts(attempts):
    base = geom.track_summary(attempts, 1.0)
    selected = [r for r in attempts if r['min_angle'] == 1 and not r.get('conflict')]
    base['computational_failure_rate'] = base.pop('attempt_failure_rate')
    base['computed_inside_image_median_px'] = base.pop('success_median_px')
    base['computed_inside_image_p90_px'] = base.pop('success_p90_px')
    for threshold in [2, 4]:
        base[f'all_attempts_under_{threshold}px_fraction'] = (
            sum(r['reason'] == 'ok' and r['error_px'] <= threshold for r in selected) / len(selected) if selected else None)
    finite = [r['error_px'] for r in selected if r.get('error_px') is not None and np.isfinite(r['error_px'])]
    base['all_finite_error_p90_px'] = float(np.percentile(finite, 90)) if finite else None
    return base


def evaluate():
    validate()
    if read(OUT / 'mask_review.json')['state'] != 'accepted':
        raise ValueError('Masks must be visually reviewed')
    frames = read(OUT / 'frames.json')['frames']
    for directory in ['features', 'matches', 'matches_overlay']:
        (OUT / directory).mkdir(exist_ok=True)
    for f in frames:
        f['K'], f['T'] = np.array(f['K']), np.array(f['T'])
        f['mask'] = geom.read_mask(f['mask'])
        f['xy'], f['descriptors'] = geom.extract_features(geom.read_image(f['image']), f['mask'])
        np.savez_compressed(OUT / 'features' / f"{f['original_frame']}.npz", xy=f['xy'], descriptors=f['descriptors'])
        print('FEATURES', f['original_frame'], len(f['xy']), flush=True)
    matches = {}
    for a, b in itertools.combinations(frames, 2):
        ia, ib = geom.mutual_ratio_matches(a['descriptors'], b['descriptors'])
        matches[(a['original_frame'], b['original_frame'])] = (ia, ib)
        np.savez_compressed(OUT / 'matches' / f"{a['original_frame']}_{b['original_frame']}.npz", idx_a=ia, idx_b=ib)
    rows, global_attempts = geom.evaluate_current(frames, matches)
    index = {f['original_frame']: f['index'] for f in frames}
    for r in rows:
        r['index_gap'] = abs(index[r['frame_a']] - index[r['frame_b']])
    geom.write_csv(OUT / 'pair_metrics.csv', rows, geom.PAIR_FIELDS + ['index_gap'])
    geom._write_overlays(OUT, frames, matches, rows)
    groups = dict(all=pair_summary(rows), adjacent=pair_summary([r for r in rows if r['index_gap']==1]))
    for lo, hi in [(0, 10), (10, 15), (15, 30), (30, 180)]:
        groups[f'{lo}-{hi}deg'] = pair_summary([r for r in rows if lo < r['azimuth_gap_deg'] <= hi])
    working = {f['original_frame']: dict(K=f['K'], T=f['T'], xy=f['xy'], mask=f['mask']) for f in frames}
    edge_sets = {}
    for (a, b), (ia, ib) in matches.items():
        if a < b:
            edge_sets[(a,b)] = set(zip(ia.tolist(), ib.tolist()))
        else:
            edge_sets[(b,a)] = set(zip(ib.tolist(), ia.tolist()))
    local_attempts, cycle_attempts, triple_rows = [], [], []
    for ti in range(6):
        ids = [f['original_frame'] for f in frames[ti:ti+3]]
        local_matches = {key: value for key, value in matches.items() if all(x in ids for x in key)}
        tracks = [t for t in geom.build_tracks(local_matches) if t['n_views'] >= 3]
        conflict = sum(t['conflict'] for t in tracks)
        raw, closed = [], []
        for t in tracks:
            t = dict(t, track_id=ti * 100000 + t['track_id'])
            attempts = []
            for threshold in [.5, 1., 2.]:
                attempts.extend(geom.heldout_attempts(t, working, threshold))
            raw.extend(attempts)
            if direct_cycle(t, edge_sets):
                closed.extend(attempts)
        local_attempts.extend(raw)
        cycle_attempts.extend(closed)
        triple_rows.append(dict(triple_index=ti, frames=ids,
            max_azimuth_gap_deg=max(geom.angular_gap(a['azimuth_deg'],b['azimuth_deg']) for a,b in itertools.combinations(frames[ti:ti+3],2)),
            raw_tracks=len(tracks), conflicts=conflict, raw=summarized_attempts(raw), cycle=summarized_attempts(closed)))
    fieldnames = ['track_id','n_views','conflict','heldout_frame','fit_a','fit_b','min_angle','parallax_deg','reason','error_px','depth_a','depth_b','depth_held']
    for label, attempts in [('global_raw', global_attempts), ('local_raw', local_attempts), ('local_cycle', cycle_attempts)]:
        geom.write_csv(OUT / f'tracks_{label}.csv', attempts, fieldnames)
    global_tracks = [t for t in geom.build_tracks(matches) if t['n_views'] >= 3]
    summary = dict(pair_groups=groups, tracks=dict(global_raw=summarized_attempts(global_attempts),
        local_raw=summarized_attempts(local_attempts), local_cycle=summarized_attempts(cycle_attempts)),
        global_raw_tracks=len(global_tracks), global_conflicts=sum(t['conflict'] for t in global_tracks),
        triples=triple_rows, camera_status='estimated background PnP, not pose ground truth')
    save(OUT / 'summary.json', summary)
    plots(rows, frames, [('global raw',global_attempts),('local raw',local_attempts),('local cycle',cycle_attempts)])
    write_report(summary)
    validate()
    save(OUT / 'audit.json', dict(frozen_files_verified=True, images=8, pairs=len(rows),
        local_triples=len(triple_rows), no_epipolar_match_prefilter=True,
        no_heldout_error_selection=True, mask_review=read(OUT/'mask_review.json'),
        metrics_review='pending'))
    save(OUT / 'status.json', dict(state='evaluated_pending_review'))
    print('SUMMARY', json.dumps(summary['pair_groups']), json.dumps(summary['tracks']), flush=True)


def plots(rows, frames, attempt_groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = [f['original_frame'] for f in frames]
    pos = {x:i for i,x in enumerate(ids)}
    values = np.full((8,8), np.nan)
    fig, ax = plt.subplots(figsize=(9,7))
    for r in rows:
        i,j=pos[r['frame_a']],pos[r['frame_b']]
        if not r['low_support']:
            values[i,j]=values[j,i]=r['epi_median_px']
    im=ax.imshow(values,cmap='viridis')
    for r in rows:
        i,j=pos[r['frame_a']],pos[r['frame_b']]
        label=f"{r['epi_median_px']:.2f}\nn={r['n_matches']}" if not r['low_support'] else f"LOW\nn={r['n_matches']}"
        for x,y in [(i,j),(j,i)]: ax.text(x,y,label,ha='center',va='center',fontsize=8,color='white' if not r['low_support'] else 'black')
    labels=[f"{f['original_frame']}\n{f['azimuth_deg']:.1f} deg" for f in frames]
    ax.set_xticks(range(8), labels, rotation=45);ax.set_yticks(range(8),labels)
    ax.set_title('Dense real GT: median sqrt(Sampson), low support masked')
    fig.colorbar(im,ax=ax,label='px at 832x480');fig.tight_layout();fig.savefig(OUT/'pair_heatmap.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name, attempts in attempt_groups:
        errors=sorted(r['error_px'] for r in attempts if r['min_angle']==1 and r.get('error_px') is not None and np.isfinite(r['error_px']))
        if errors:ax.plot(np.maximum(errors,1e-4),np.arange(1,len(errors)+1)/len(errors),label=f'{name}, n={len(errors)}')
    ax.set_xscale('log');ax.set_xlabel('Held-out reprojection error (px), including outside-image finite errors')
    ax.set_ylabel('Empirical cumulative fraction');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'track_error_cdf.png',dpi=140);plt.close(fig)


def write_report(summary):
    text=['# 密集真实八帧几何一致性评估','',
        '相机来自背景PnP定位到原真实Omega重建世界，RGB是真实视频，位姿不是标定真值。未运行生成图评估。',
        '沿用real8的SIFT12000、双向ratio0.75和mutual；主指标不做极线预筛；匹配<30单列低支持。误差为832×480像素。', '',
        '| 配对组 | 对数 | 有支持 | 支持组极线中位数 | 支持组P90的中位数 | 匹配数中位数 |','|---|---:|---:|---:|---:|---:|']
    for label,g in summary['pair_groups'].items():
        text.append(f"| {label} | {g['n_pairs']} | {g['supported_pairs']} | {g['supported_median_pair_epi_px']} | {g['supported_median_pair_p90_px']} | {g['median_matches']} |")
    text+=['','| 轨迹组 | 无冲突轨迹 | 检查数 | 可计算且在图内误差中位数 | P90 | 计算失败率 | 全轨迹<4px比例 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,s in summary['tracks'].items():
        text.append(f"| {label} | {s['n_tracks']} | {s['n_attempts']} | {s['computed_inside_image_median_px']} | {s['computed_inside_image_p90_px']} | {s['computational_failure_rate']} | {s['valid_track_fraction_4px']} |")
    text+=['','局部组三视角是按方位排序的六个连续三元组。cycle组要求三条两两描述子匹配直接闭合，筛选不使用相机极线或留出投影误差；它是原始局部组的补充，不能替代原始结果。',
        '计算成功仅表示可计算、正深度、满足视差且投影在图内，不等于误差小。2/4px仅描述分布，不是合格阈值。',
        '三元组相互重叠，轨迹/留出检查并非独立样本。当前GT车体像素大小、拍摄轨迹及相机来源与先前real8不同，不能把数值变化全归因于采样密度。',
        '逐对数据见pair_metrics.csv；三类原始留出检查见tracks_*.csv；summary.json含六个三元组的分项结果和计算失败原因。','']
    (OUT/'REPORT.md').write_text('\n'.join(text))


if __name__ == '__main__':
    cv2.setNumThreads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','evaluate'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else evaluate()
