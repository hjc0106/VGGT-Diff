"""Same-camera real/generated geometry comparison with fixed support accounting."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

import crossview_geometry_eval as geom
import crossview_geometry_dense_gt as dense
import vggtdiff_matched_gt as job
from vggtdiff_white_car import REPO, digest, save
from vggtdiff_latest_coverage_ab import montage

OUT = job.OUT
EVAL = OUT / 'evaluation'
TRACK_FIELDS = ['track_id','n_views','conflict','heldout_frame','fit_a','fit_b','min_angle','parallax_deg','reason','error_px','depth_a','depth_b','depth_held','triple_index']


def read(path):
    return json.loads(path.read_text())


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def key(row):
    return tuple(sorted([int(row['frame_a']),int(row['frame_b'])]))


def paired_deltas(gt_rows, gen_rows, fixed_pairs):
    gt, gen = {key(r):r for r in gt_rows}, {key(r):r for r in gen_rows}
    rows=[]
    for pair in fixed_pairs:
        ordered=tuple(pair);pair=tuple(sorted(pair));a,b=gt[pair],gen[pair]
        both = not a['low_support'] and not b['low_support']
        va,vb=a.get('epi_median_px'),b.get('epi_median_px')
        rows.append(dict(frame_a=ordered[0],frame_b=ordered[1],gt_matches=a['n_matches'],generated_matches=b['n_matches'],
            gt_supported=not a['low_support'],generated_supported=not b['low_support'],joint_supported=both,
            lost_support=not a['low_support'] and b['low_support'],
            gt_epi_px=va,generated_epi_px=vb,delta_epi_px=(vb-va) if va is not None and vb is not None else None,
            gt_p90_px=a.get('epi_p90_px'),generated_p90_px=b.get('epi_p90_px'),
            gt_coverage_a=a['coverage_a'],gt_coverage_b=a['coverage_b'],
            generated_coverage_a=b['coverage_a'],generated_coverage_b=b['coverage_b']))
    supported=[r for r in rows if r['joint_supported']]
    return rows, dict(n_fixed=len(rows),gt_supported=sum(r['gt_supported'] for r in rows),
        generated_supported=sum(r['generated_supported'] for r in rows),joint_supported=len(supported),
        lost_gt_support=sum(r['lost_support'] for r in rows),
        joint_median_gt_epi_px=median([r['gt_epi_px'] for r in supported]),
        joint_median_generated_epi_px=median([r['generated_epi_px'] for r in supported]),
        joint_median_paired_delta_px=median([r['delta_epi_px'] for r in supported]),
        raw_all_fixed_median_delta_px=median([r['delta_epi_px'] for r in rows]),
        joint_median_gt_coverage=median([min(r['gt_coverage_a'],r['gt_coverage_b']) for r in supported]),
        joint_median_generated_coverage=median([min(r['generated_coverage_a'],r['generated_coverage_b']) for r in supported]))


def freeze_evaluator():
    job.validate()
    EVAL.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__).resolve(),Path(geom.__file__).resolve(),Path(dense.__file__).resolve()]
    manifest=dict(frozen_files=[dict(path=str(p),sha256=digest(p)) for p in files],
        protocol_sha256=digest(OUT/'protocol.json'))
    if (EVAL/'implementation.json').exists() and read(EVAL/'implementation.json')!=manifest:
        raise ValueError('Frozen evaluator differs')
    save(EVAL/'implementation.json',manifest)
    for p in files:shutil.copyfile(p,EVAL/p.name)


def validate_evaluator():
    job.validate();job.verify_rows(read(EVAL/'implementation.json')['frozen_files'])
    assert read(EVAL/'implementation.json')['protocol_sha256']==digest(OUT/'protocol.json')


def segment(seeds):
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    from vggtdiff_car15_groupa_prepare import choose_mask
    job.validate()
    segmenter=SAM3Segmenter(SimpleNamespace(source_dir=str(REPO/'thirdparty/sam3'),
        checkpoint_path=str(REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
        device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
    for seed in seeds:
        root=OUT/f'seed_{seed}';result=read(root/'result.json');job.verify_rows(result['frames'])
        (root/'masks').mkdir(exist_ok=True)
        overlays=[];stats=[]
        for i,r in enumerate(result['frames']):
            rgb=np.asarray(Image.open(r['path']).convert('RGB'));path=root/'masks'/f'{i:02d}.png'
            if not path.exists():
                mask=choose_mask(segmenter,rgb)
                Image.fromarray(mask.astype('uint8')*255).save(path)
            mask=geom.read_mask(path)>0
            assert mask.shape==(480,832) and mask.sum()>1000
            over=rgb.copy();over[mask]=(.55*rgb[mask]+.45*np.array([0,255,0])).astype('uint8')
            cv2.drawContours(over,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,0,0),1)
            dest=root/'masks'/f'{i:02d}_overlay.jpg';Image.fromarray(over).save(dest);overlays.append(dest)
            stats.append(dict(index=i,path=str(path),sha256=digest(path),area=int(mask.sum())))
            print('MASK',seed,i,int(mask.sum()),flush=True)
        montage(overlays,[f"{i:02d} | {result['frames'][i]['angle']:.2f} deg" for i in range(8)],root/'mask_review.jpg',columns=4)
        save(root/'mask_stats.json',dict(frames=stats))
        save(root/'status.json',dict(state='segmented_pending_review'))


def feature_frame(meta, directory, cached=None):
    r=dict(meta);r['K']=np.asarray(r['K'],float);r['T']=np.asarray(r['T'],float)
    r['mask']=geom.read_mask(r['mask'])
    directory.mkdir(parents=True,exist_ok=True)
    if cached is not None:
        with np.load(cached) as d:r['xy'],r['descriptors']=d['xy'],d['descriptors']
    else:r['xy'],r['descriptors']=geom.extract_features(geom.read_image(r['image']),r['mask'])
    np.savez_compressed(directory/f"{r['original_frame']}.npz",xy=r['xy'],descriptors=r['descriptors'])
    return r


def load_targets(seed=None):
    meta=read(OUT/'targets.json')['frames']
    root=EVAL/'gt' if seed is None else EVAL/f'seed_{seed}'
    if seed is not None:
        generated=read(OUT/f'seed_{seed}/result.json');job.verify_rows(generated['frames'])
        assert generated['cameras_sha256']==digest(OUT/'trajectory.json')
        assert read(OUT/f'seed_{seed}/visual_review.json')['state']=='accepted'
        masks=read(OUT/f'seed_{seed}/mask_stats.json')['frames'];job.verify_rows(masks)
    frames=[]
    for i,r in enumerate(meta):
        r=dict(r)
        if seed is not None:
            assert generated['frames'][i]['original_frame']==r['original_frame']
            r['image']=generated['frames'][i]['path'];r['mask']=masks[i]['path']
        cached=job.BASELINE/'features'/f"{r['original_frame']}.npz" if seed is None else None
        frames.append(feature_frame(r,root/'features',cached))
    return frames,root


def build_matches(frames,root,reuse_gt=False):
    (root/'matches').mkdir(exist_ok=True)
    matches={}
    for a,b in itertools.combinations(frames,2):
        ids=(a['original_frame'],b['original_frame'])
        if reuse_gt:
            with np.load(job.BASELINE/'matches'/f'{ids[0]}_{ids[1]}.npz') as d:
                ia,ib=d['idx_a'],d['idx_b']
        else:ia,ib=geom.mutual_ratio_matches(a['descriptors'],b['descriptors'])
        matches[ids]=(ia,ib)
        np.savez_compressed(root/'matches'/f'{ids[0]}_{ids[1]}.npz',idx_a=ia,idx_b=ib)
    return matches


def assess_targets(frames,root,reuse_gt=False):
    matches=build_matches(frames,root,reuse_gt)
    rows,global_attempts=geom.evaluate_current(frames,matches)
    index={f['original_frame']:i for i,f in enumerate(frames)}
    for r in rows:r['index_gap']=abs(index[r['frame_a']]-index[r['frame_b']])
    geom.write_csv(root/'pair_metrics.csv',rows,geom.PAIR_FIELDS+['index_gap'])
    working={f['original_frame']:dict(K=f['K'],T=f['T'],xy=f['xy'],mask=f['mask']) for f in frames}
    edges={}
    for (a,b),(ia,ib) in matches.items():
        edges[tuple(sorted((a,b)))]=set(zip(ia.tolist(),ib.tolist())) if a<b else set(zip(ib.tolist(),ia.tolist()))
    raw_all,cycle_all,triples=[],[],[]
    for ti in range(6):
        ids=[f['original_frame'] for f in frames[ti:ti+3]]
        sub={k:v for k,v in matches.items() if all(x in ids for x in k)}
        tracks=[t for t in geom.build_tracks(sub) if t['n_views']>=3]
        raw,cycle=[],[]
        for track in tracks:
            track=dict(track,track_id=ti*100000+track['track_id'])
            attempts=[]
            for angle in [.5,1.,2.]:attempts+=geom.heldout_attempts(track,working,angle)
            for a in attempts:a['triple_index']=ti
            raw+=attempts
            if dense.direct_cycle(track,edges):cycle+=attempts
        raw_all+=raw;cycle_all+=cycle
        triples.append(dict(frames=ids,raw=dense.summarized_attempts(raw),cycle=dense.summarized_attempts(cycle)))
    for name,attempts in [('global_raw',global_attempts),('local_raw',raw_all),('local_cycle',cycle_all)]:
        geom.write_csv(root/f'tracks_{name}.csv',attempts,TRACK_FIELDS)
    summary=dict(pair_all=dense.pair_summary(rows),local_raw=dense.summarized_attempts(raw_all),
        local_cycle=dense.summarized_attempts(cycle_all),global_raw=dense.summarized_attempts(global_attempts),
        subset6_local_raw=dense.summarized_attempts([a for a in raw_all if a['triple_index']>=2]),
        subset6_local_cycle=dense.summarized_attempts([a for a in cycle_all if a['triple_index']>=2]),triples=triples)
    save(root/'geometry_summary.json',summary)
    return rows,summary


def load_references():
    return [feature_frame(r,EVAL/'references/features') for r in read(OUT/'references.json')['frames']]


def reference_metrics(targets,references,root):
    rows=[]
    (root/'reference_matches').mkdir(exist_ok=True)
    for a in targets:
        for b in references:
            ia,ib=geom.mutual_ratio_matches(a['descriptors'],b['descriptors'])
            r=geom.pair_record(a['original_frame'],b['original_frame'],a['xy'][ia],b['xy'][ib],
                a['mask'],b['mask'],a['K'],a['T'],b['K'],b['T'])
            r['azimuth_gap_deg']=geom.angular_gap(a['azimuth_deg'],b['azimuth_deg'])
            rows.append(r)
            np.savez_compressed(root/'reference_matches'/f"{a['original_frame']}_{b['original_frame']}.npz",idx_a=ia,idx_b=ib)
    geom.write_csv(root/'reference_metrics.csv',rows,geom.PAIR_FIELDS)
    return rows


def mask_distances(a,b):
    a,b=a>0,b>0
    union=(a|b).sum();iou=float((a&b).sum()/union) if union else None
    aa=cv2.morphologyEx(a.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    bb=cv2.morphologyEx(b.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    if not aa.any() or not bb.any():return iou,None,None
    da=cv2.distanceTransform((~aa).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    db=cv2.distanceTransform((~bb).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    values=np.r_[da[bb],db[aa]]
    return iou,float(values.mean()),float(np.percentile(values,90))


def same_view_metrics(gt,gen,root):
    rows=[]
    for a,b in zip(gt,gen):
        assert np.array_equal(a['K'],b['K']) and np.array_equal(a['T'],b['T'])
        ia,ib=geom.mutual_ratio_matches(a['descriptors'],b['descriptors'])
        values=np.linalg.norm(a['xy'][ia]-b['xy'][ib],axis=1)
        iou,mean,p90=mask_distances(a['mask'],b['mask'])
        rows.append(dict(frame=a['original_frame'],index=a['index'],n_matches=len(ia),low_support=len(ia)<30,
            displacement_median_px=median(values),displacement_p90_px=float(np.percentile(values,90)) if len(values) else None,
            gt_coverage=geom.coverage_fraction(a['xy'][ia],a['mask']),generated_coverage=geom.coverage_fraction(b['xy'][ib],b['mask']),
            mask_iou=iou,boundary_mean_px=mean,boundary_p90_px=p90))
        np.savez_compressed(root/f"same_view_{a['original_frame']}.npz",gt_xy=a['xy'][ia],generated_xy=b['xy'][ib])
    geom.write_csv(root/'same_view_metrics.csv',rows,list(rows[0]))
    return dict(n_views=8,supported_views=sum(not r['low_support'] for r in rows),
        median_mask_iou=median([r['mask_iou'] for r in rows]),
        median_boundary_mean_px=median([r['boundary_mean_px'] for r in rows]),
        supported_median_displacement_px=median([r['displacement_median_px'] for r in rows if not r['low_support']]),
        all_view_median_displacement_px=median([r['displacement_median_px'] for r in rows]),rows=rows)


def evaluate(seeds):
    validate_evaluator()
    gt,gt_root=load_targets();references=load_references()
    gt_rows,gt_summary=assess_targets(gt,gt_root,reuse_gt=True)
    old=read(job.BASELINE/'summary.json')['tracks']
    for group in ['global_raw','local_raw','local_cycle']:
        for metric in ['computed_inside_image_median_px','computed_inside_image_p90_px','valid_track_fraction_4px']:
            assert np.isclose(gt_summary[group][metric],old[group][metric],atol=1e-10), (group,metric)
    gt_reference=reference_metrics(gt,references,gt_root)
    protocol=read(OUT/'protocol.json')
    primary=protocol['primary_pairs'];adjacent=protocol['adjacent_pairs']
    secondary=[p for p in primary if not set(p)&{1340,1300}]
    ref_primary=[[r['frame_a'],r['frame_b']] for r in gt_reference if r['azimuth_gap_deg']<=15]
    summaries={}
    for seed in seeds:
        gen,root=load_targets(seed);rows,geometry=assess_targets(gen,root)
        cross=reference_metrics(gen,references,root)
        comparisons={}
        for name,fixed in [('primary14',primary),('adjacent7',adjacent),('subset6_primary',secondary)]:
            delta,summary=paired_deltas(gt_rows,rows,fixed)
            geom.write_csv(root/f'delta_{name}.csv',delta,list(delta[0]))
            comparisons[name]=summary
        for name,fixed in [('reference_all',[[r['frame_a'],r['frame_b']] for r in gt_reference]),
                           ('reference_near',ref_primary),
                           ('subset6_reference_near',[p for p in ref_primary if p[0] not in [1340,1300]])]:
            delta,summary=paired_deltas(gt_reference,cross,fixed)
            geom.write_csv(root/f'delta_{name}.csv',delta,list(delta[0]))
            comparisons[name]=summary
        same=same_view_metrics(gt,gen,root)
        result=dict(seed=seed,paired=comparisons,geometry=geometry,same_view=same)
        save(root/'summary.json',result);summaries[str(seed)]=result
        save(root/'audit.json',dict(gt_geometry_reproduced=True,cameras_exactly_equal=True,
            generated_image_hashes_verified=True,mask_hashes_verified=True,no_epipolar_prefilter=True,
            fixed_primary_pairs=len(primary),fixed_local_triples=6,heldout_error_not_used_for_filtering=True))
        print('EVALUATED',seed,json.dumps(comparisons['primary14']),flush=True)
    save(EVAL/'gt/summary.json',dict(geometry=gt_summary,reference_primary_pairs=ref_primary))
    validate_evaluator()


def report():
    validate_evaluator()
    data={str(seed):read(EVAL/f'seed_{seed}/summary.json') for seed in job.SEEDS}
    gt=read(EVAL/'gt/summary.json')
    save(OUT/'comparison_summary.json',dict(gt=gt,seeds=data,
        caveat='Estimated cameras; three seeds and overlapping pairs are descriptive, not independent significance tests.'))
    lines=['# 同相机生成图与真实GT几何对照','',
        '六张真实条件图固定，八个目标与真实GT使用完全相同的K和W2C。NVS、832×480、50步，三个预先固定种子。相机仍为估计值。',
        'SIFT12000、双向ratio0.75、mutual，主指标不使用极线预筛；支持线30仅用于证据分层。低支持配对未记为零、未从固定集合静默删除。','',
        '## 固定14对近距配对','',
        '| seed | GT/生成支持 | 同时有支持 | 支持丢失 | GT极线中位数 | 生成极线中位数 | 逐对差值中位数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for seed,s in data.items():
        a=s['paired']['primary14'];lines.append(f"| {seed} | {a['gt_supported']}/{a['generated_supported']} | {a['joint_supported']} | {a['lost_gt_support']} | {a['joint_median_gt_epi_px']} | {a['joint_median_generated_epi_px']} | {a['joint_median_paired_delta_px']} |")
    lines+=['','极线数值在同一组双方均有支持的配对上计算；完整固定14对及覆盖率见每个种子的delta_primary14.csv。',
        '', '## 相邻配对与六帧敏感性分析','',
        '| seed | 相邻7对增量 | 排除1340/1300后固定配对数 | 子集GT/生成支持 | 子集极线增量 |',
        '|---|---:|---:|---:|---:|']
    for seed,s in data.items():
        a=s['paired']['adjacent7'];b=s['paired']['subset6_primary']
        lines.append(f"| {seed} | {a['joint_median_paired_delta_px']} | {b['n_fixed']} | {b['gt_supported']}/{b['generated_supported']} | {b['joint_median_paired_delta_px']} |")
    lines += ['',
        '', '## 三视角（局部描述子闭环）','',
        '| 组 | 轨迹数 | 图内可计算误差中位数 | P90 | 计算失败率 | 全轨迹<4px比例 |','|---|---:|---:|---:|---:|---:|']
    for name,g in [('GT',gt['geometry']['local_cycle'])]+[(seed,s['geometry']['local_cycle']) for seed,s in data.items()]:
        lines.append(f"| {name} | {g['n_tracks']} | {g['computed_inside_image_median_px']} | {g['computed_inside_image_p90_px']} | {g['computational_failure_rate']} | {g['valid_track_fraction_4px']} |")
    lines+=['','原始局部轨迹、全局原始轨迹及排除1340/1300的六帧敏感性结果全部保留于各seed的summary.json和tracks_*.csv。闭环仅使用描述子直接三边支持，未按待测重投影误差筛选。',
        '', '## 与真实参考的兼容性','',
        '| seed | 固定近距配对数 | GT/生成支持 | 逐对极线增量中位数 |','|---|---:|---:|---:|']
    for seed,s in data.items():
        a=s['paired']['reference_near'];lines.append(f"| {seed} | {a['n_fixed']} | {a['gt_supported']}/{a['generated_supported']} | {a['joint_median_paired_delta_px']} |")
    lines+=['','## 同视角外观结构对照','',
        '| seed | 匹配充分视角 | 描述子位置偏差中位数 | 掩码IoU中位数 | 对称边界均距中位数 |','|---|---:|---:|---:|---:|']
    for seed,s in data.items():
        a=s['same_view'];lines.append(f"| {seed} | {a['supported_views']} | {a['supported_median_displacement_px']} | {a['median_mask_iou']} | {a['median_boundary_mean_px']} |")
    lines+=['','同视角对应未做单应性或其它配准；位置偏差含描述子误匹配与外观变化影响，掩码边界受分割误差影响，仅作补充。',
        '低极线误差不充分证明几何真实，特别是深度/沿极线变化。较低于GT的误差也可能来自平滑纹理、不同匹配支持或GT位姿误差，必须结合跨真实参考和同视角结构对照解释。',
        '重复种子不等于独立场景，不报告匹配点级显著性。所有像素指标均为832×480栅格；未做高斯重建。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    plots(data)
    paths=[];labels=[]
    for i in range(8):
        paths.append(job.GT/'gt'/f'{i:02d}.png');labels.append(f'GT {i:02d}')
        for seed in job.SEEDS:
            paths.append(OUT/f'seed_{seed}/raw'/f'{i:02d}.png');labels.append(f'seed {seed}')
    montage(paths,labels,OUT/'gt_vs_three_seeds.jpg',columns=4)
    import imageio.v2 as imageio
    with imageio.get_writer(str(OUT/'gt_vs_three_seeds.mp4'),fps=2,macro_block_size=16) as writer:
        for i in range(8):
            writer.append_data(np.concatenate([np.asarray(Image.open(paths[i*4+j])) for j in range(4)],axis=1))
    save(OUT/'status.json',dict(state='compared_pending_final_review',seeds=job.SEEDS))


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed,s in data.items():
        import csv
        rows=list(csv.DictReader((EVAL/f'seed_{seed}/delta_primary14.csv').open()))
        x=np.arange(len(rows))
        y=[float(r['delta_epi_px']) if r['joint_supported']=='True' and r['delta_epi_px'] else np.nan for r in rows]
        axes[0].plot(x,y,marker='o',label=seed)
        a=s['paired']['reference_near'];axes[1].scatter(seed,a['joint_median_paired_delta_px'],label=seed)
    axes[0].axhline(0,color='black',linewidth=.8);axes[0].set_xlabel('Fixed close pair index');axes[0].set_ylabel('Generated minus GT sqrt(Sampson), px')
    axes[0].set_title('Fourteen preselected pairs; unsupported omitted from line only');axes[0].legend()
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set_title('Target-to-real-reference paired delta');axes[1].set_ylabel('Median paired delta, px')
    fig.tight_layout();fig.savefig(OUT/'paired_deltas.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    cv2.setNumThreads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','segment','evaluate','report'])
    parser.add_argument('--seeds',nargs='+',type=int,default=job.SEEDS)
    args=parser.parse_args()
    if args.action=='freeze':freeze_evaluator()
    elif args.action=='segment':segment(args.seeds)
    elif args.action=='evaluate':evaluate(args.seeds)
    else:report()
