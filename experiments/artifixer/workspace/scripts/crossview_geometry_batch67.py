"""Vehicle-only fixed-camera geometry audit across generated batches6 and7.

prepare -> visually inspect masks -> accept-masks -> evaluate.
Frozen SIFT/mutual-ratio, epipolar and held-out-triple helpers are reused verbatim.
No epipolar/RANSAC match filtering, camera fitting or reprojection-error selection.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import crossview_geometry_eval as geom
import crossview_geometry_dense_gt as dense
import crossview_geometry_matched_eval as matched
import vggtdiff_batch07_continuous as job

BASE=job.chain.BASE
B6=BASE/'crossview_geometry_matched_gt_v1/seed_20261028'
B7=job.DEFAULT_OUTPUT
OUT=BASE/'crossview_geometry_batch06_batch07_continuous_v1'
TRACK_FIELDS=matched.TRACK_FIELDS+['group']


def read(path):return job.chain.read(path)
def save(path,value):return job.chain.save(path,value)
def frozen(path,value):return job.write_frozen_json(path,value)


def prepare():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    from vggtdiff_car15_groupa_prepare import choose_mask
    torch.set_num_threads(4);cv2.setNumThreads(4)
    job.chain.validate(B7)
    t6,t7=read(B6.parent/'trajectory.json'),read(B7/'trajectory.json')
    r6,r7=read(B6/'result.json'),read(B7/'result.json')
    assert r6['cameras_sha256']==job.chain.digest(B6.parent/'trajectory.json')
    assert r7['cameras_sha256']==job.chain.digest(B7/'trajectory.json')
    assert read(B6/'visual_review.json')['state']==read(B7/'visual_review.json')['state']=='accepted'
    rows=[]
    for batch,root,result,t in [(6,B6,r6,t6),(7,B7,r7,t7)]:
        for i,r in enumerate(result['frames']):
            image=Path(r['path']);assert job.chain.digest(image)==r['sha256']
            assert np.isclose(r['angle'],t['target_angles_deg'][i])
            ident=i if batch==6 else 8+i
            rows.append(dict(index=ident,original_frame=ident,batch=batch,local_index=i,
                label=f'B{batch}_{i:02d}',image=str(image),sha256=r['sha256'],
                azimuth_deg=r['angle'],K=t['target_intrinsics'][i],T=t['target_w2c'][i],
                mask=str(OUT/'masks'/f'{ident:02d}.png'),
                video_frame=r.get('original_frame'),orbit_index=r['index'] if batch==7 else None,
                conditioning_reference_for_batch7=(batch==6 and i in read(B7/'manifest.json')['source_indices'])))
    assert len(rows)==16
    specs=[]
    for a,b in itertools.combinations(rows,2):
        specs.append(dict(frame_a=a['index'],frame_b=b['index'],label_a=a['label'],label_b=b['label'],
            group='cross_batch' if a['batch']!=b['batch'] else f"batch{a['batch']}",
            index_gap=b['index']-a['index'],azimuth_gap_deg=geom.angular_gap(a['azimuth_deg'],b['azimuth_deg'])))
    protocol=dict(image_size=[832,480],source_run_batch6=str(B6),source_run_batch7=str(B7),
        matcher=dict(type='SIFT',nfeatures=12000,mutual=True,ratio=.75,epipolar_prefilter=False,RANSAC=False),
        region='Vehicle SAM3 masks; batch6 frozen masks reused; batch7 same existing SAM3 white-car selector; review before metrics.',
        support_minimum_matches=30,spatial_coverage_grid=[8,6],
        primary_pair_rule='All pairs <=15deg, grouped by batch6/batch7/cross_batch, fixed before metrics.',
        secondary_pair_rule='All120 pairs plus adjacent pairs and <=30deg sensitivity; no unsupported pair silently discarded.',
        primary_cross_pairs=[[p['frame_a'],p['frame_b']] for p in specs if p['group']=='cross_batch' and p['azimuth_gap_deg']<=15],
        triple_rule='All14 consecutive local triples; report batch6 six, boundary two, batch7 six separately. Raw and direct three-edge descriptor-cycle tracks.',
        parallax_degrees=[.5,1,2],primary_parallax_degrees=1,
        error_selection=False,camera_optimization=False,acceptance_thresholds=None,
        camera_status='Existing estimated batch6 GT-aligned cameras and requested rigidly continued batch7 cameras, not measured generated-camera GT.',
        interpretation='Conditional geometry compatibility with specified cameras; not complete3D accuracy or proof of no floaters. Batch7 has no same-camera real GT. Primary cross pairs all involve prior conditioning references.')
    frozen(OUT/'protocol.json',protocol);frozen(OUT/'pair_spec.json',dict(pairs=specs))
    overlays=[];segmenter=None
    mask_stats={r['index']:r for r in read(B6/'mask_stats.json')['frames']}
    for row in rows:
        i=row['index'];path=Path(row['mask']);path.parent.mkdir(parents=True,exist_ok=True)
        if row['batch']==6:
            source=Path(mask_stats[row['local_index']]['path'])
            assert job.chain.digest(source)==mask_stats[row['local_index']]['sha256']
            job.chain.frozen_copy(source,path)
        elif not path.exists():
            if segmenter is None:
                segmenter=SAM3Segmenter(SimpleNamespace(source_dir=str(job.chain.REPO/'thirdparty/sam3'),
                    checkpoint_path=str(job.chain.REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
                    device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
            rgb=np.asarray(Image.open(row['image']).convert('RGB'));mask=choose_mask(segmenter,rgb)
            Image.fromarray(mask.astype('uint8')*255).save(path)
        mask=geom.read_mask(path)>0;ys,xs=np.where(mask)
        assert mask.shape==(480,832) and len(xs)>1000
        row.update(mask_sha256=job.chain.digest(path),mask_bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],mask_area=int(mask.sum()))
        rgb=np.asarray(Image.open(row['image']).convert('RGB'));over=rgb.copy()
        over[mask]=(.55*rgb[mask]+.45*np.array([0,255,0])).astype('uint8')
        cv2.drawContours(over,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,0,0),1)
        dest=OUT/'masks'/f'{i:02d}_overlay.jpg';Image.fromarray(over).save(dest);overlays.append(dest)
        print('MASK',row['label'],row['mask_area'],flush=True)
    del segmenter;torch.cuda.empty_cache()
    frozen(OUT/'frames.json',dict(frames=rows))
    job.chain.montage(overlays,[f"{r['label']} | {r['azimuth_deg']:.2f} deg" for r in rows],OUT/'mask_review.jpg',columns=4)
    files=[Path(__file__).resolve(),Path(geom.__file__).resolve(),Path(dense.__file__).resolve(),Path(matched.__file__).resolve(),
        OUT/'protocol.json',OUT/'pair_spec.json',OUT/'frames.json',B6/'result.json',B6.parent/'trajectory.json',
        B6/'mask_stats.json',B7/'result.json',B7/'trajectory.json',B7/'manifest.json']
    files += [Path(r[key]) for r in rows for key in ['image','mask']]
    frozen(OUT/'manifest.json',dict(files=[dict(path=str(p),sha256=job.chain.digest(p)) for p in files]))
    for p in [Path(__file__),Path(geom.__file__),Path(dense.__file__),Path(matched.__file__)]:
        job.chain.frozen_copy(p.resolve(),OUT/'implementation'/p.name)
    save(OUT/'status.json',dict(state='prepared_pending_mask_review'))
    print('PREPARED',OUT,flush=True)


def validate():
    for r in read(OUT/'manifest.json')['files']:
        assert job.chain.digest(Path(r['path']))==r['sha256'],r['path']


def accept_masks():
    validate()
    save(OUT/'mask_review.json',dict(state='accepted',scope='All16 vehicle masks visually inspected; windows/wheels included; road/background excluded.'))


def evaluate():
    validate();assert read(OUT/'mask_review.json')['state']=='accepted'
    cv2.setNumThreads(4)
    meta=read(OUT/'frames.json')['frames'];frames=[]
    for row in meta:
        cached=B6.parent/'evaluation/seed_20261028/features'/f"{row['video_frame']}.npz" if row['batch']==6 else None
        f=matched.feature_frame(row,OUT/'features',cached=cached);frames.append(f)
        print('FEATURES',row['label'],len(f['xy']),flush=True)
    matches=matched.build_matches(frames,OUT)
    rows,global_attempts=geom.evaluate_current(frames,matches)
    by_id={r['index']:r for r in meta}
    for r in rows:
        a,b=by_id[r['frame_a']],by_id[r['frame_b']]
        r.update(group='cross_batch' if a['batch']!=b['batch'] else f"batch{a['batch']}",
            label_a=a['label'],label_b=b['label'],index_gap=b['index']-a['index'])
    fields=geom.PAIR_FIELDS+['group','label_a','label_b','index_gap']
    geom.write_csv(OUT/'pair_metrics.csv',rows,fields)
    geom._write_overlays(OUT,frames,matches,rows)
    summaries={}
    for group in ['batch6','batch7','cross_batch']:
        sub=[r for r in rows if r['group']==group]
        summaries[group]=dict(all=dense.pair_summary(sub),primary15=dense.pair_summary([r for r in sub if r['azimuth_gap_deg']<=15]),
            sensitivity30=dense.pair_summary([r for r in sub if r['azimuth_gap_deg']<=30]),adjacent=dense.pair_summary([r for r in sub if r['index_gap']==1]))
    boundary=next(r for r in rows if (r['frame_a'],r['frame_b'])==(7,8))
    edges={k:set(zip(v[0].tolist(),v[1].tolist())) for k,v in matches.items()}
    working={f['index']:dict(K=f['K'],T=f['T'],xy=f['xy'],mask=f['mask']) for f in frames}
    attempts_by_group={g:dict(raw=[],cycle=[]) for g in ['batch6','boundary','batch7']}
    triples=[]
    for ti in range(14):
        ids=list(range(ti,ti+3));group='batch6' if ti<=5 else ('boundary' if ti in [6,7] else 'batch7')
        sub={k:v for k,v in matches.items() if set(k).issubset(ids)}
        tracks=[t for t in geom.build_tracks(sub) if t['n_views']>=3];raw=[];cycle=[]
        for track in tracks:
            track=dict(track,track_id=ti*100000+track['track_id']);attempts=[]
            for threshold in [.5,1,2]:attempts+=geom.heldout_attempts(track,working,threshold)
            for a in attempts:a.update(triple_index=ti,group=group)
            raw+=attempts
            if dense.direct_cycle(track,edges):cycle+=attempts
        attempts_by_group[group]['raw']+=raw;attempts_by_group[group]['cycle']+=cycle
        triples.append(dict(index=ti,group=group,frames=ids,labels=[by_id[i]['label'] for i in ids],
            raw_tracks=len(tracks),conflicts=sum(t['conflict'] for t in tracks),raw=dense.summarized_attempts(raw),cycle=dense.summarized_attempts(cycle)))
    track_groups={g:{key:dense.summarized_attempts(values) for key,values in v.items()} for g,v in attempts_by_group.items()}
    for key in ['raw','cycle']:
        attempts=[a for v in attempts_by_group.values() for a in v[key]]
        geom.write_csv(OUT/f'tracks_local_{key}.csv',attempts,TRACK_FIELDS)
    geom.write_csv(OUT/'tracks_global_raw.csv',global_attempts,TRACK_FIELDS)
    # Exact batch6 reproduction catches changes of feature/mask/matching conventions.
    previous=read(B6.parent/'evaluation/seed_20261028/summary.json')['paired']['primary14']['joint_median_generated_epi_px']
    current=summaries['batch6']['primary15']['supported_median_pair_epi_px']
    assert np.isclose(previous,current,atol=1e-10),(previous,current)
    stats=dict(pair_groups=summaries,boundary_pair={k:v for k,v in boundary.items() if k!='errors'},
        track_groups=track_groups,triples=triples,total_pairs=len(rows),
        low_support_pairs=sum(r['low_support'] for r in rows),
        real_batch6_baseline=read(BASE/'crossview_geometry_dense_real8_v1/summary.json')['tracks']['local_cycle'],
        batch6_metric_reproduced=True)
    save(OUT/'summary.json',stats)
    primary_cross=[r for r in rows if r['group']=='cross_batch' and r['azimuth_gap_deg']<=15]
    cross_paths=[OUT/'matches_overlay'/f"{r['frame_a']}_{r['frame_b']}.jpg" for r in primary_cross]
    job.chain.montage(cross_paths,[f"{r['label_a']} -> {r['label_b']} | n={r['n_matches']} | median {r['epi_median_px']:.3f}px" for r in primary_cross],OUT/'cross_primary_matches.jpg',columns=1,tile=(1248,360))
    plots(rows,meta)
    validate()
    save(OUT/'audit.json',dict(state='passed',frozen_inputs_verified=True,images=16,pairs=len(rows),local_triples=len(triples),
        primary_cross_pairs=len(primary_cross),batch6_primary_epi_reproduced_px=current,no_epipolar_or_ransac_match_prefilter=True,
        no_camera_refit=True,no_heldout_error_selection=True,masks_reviewed=True,geometry_quality_acceptance_threshold=None))
    save(OUT/'status.json',dict(state='evaluated_pending_interpretation'))
    print('SUMMARY',summaries,track_groups,flush=True)


def plots(rows,meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=np.full((16,16),np.nan);counts=np.zeros((16,16),int)
    for r in rows:
        a,b=r['frame_a'],r['frame_b'];counts[a,b]=counts[b,a]=r['n_matches']
        if not r['low_support']:values[a,b]=values[b,a]=r['epi_median_px']
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    a=axes[0].imshow(values,vmin=0,vmax=5,cmap='magma');fig.colorbar(a,ax=axes[0],label='Median sqrt(Sampson), px (color clipped at5)')
    axes[0].set_title('Vehicle epipolar error | <30 matches masked')
    a=axes[1].imshow(counts,cmap='viridis');fig.colorbar(a,ax=axes[1],label='Mutual descriptor matches')
    axes[1].set_title('Match support (all pairs)')
    for ax in axes:
        ax.axhline(7.5,color='cyan',linewidth=1);ax.axvline(7.5,color='cyan',linewidth=1)
        ax.set_xticks(range(16),[r['label'] for r in meta],rotation=90);ax.set_yticks(range(16),[r['label'] for r in meta])
    fig.tight_layout();fig.savefig(OUT/'pair_heatmaps.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','accept-masks','evaluate'])
    a=p.parse_args();{'prepare':prepare,'accept-masks':accept_masks,'evaluate':evaluate}[a.action]()

if __name__=='__main__':main()
