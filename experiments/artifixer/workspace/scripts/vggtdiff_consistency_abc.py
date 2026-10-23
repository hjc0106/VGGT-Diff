"""Frozen real-camera A/B/C experiment, plus two-camera held-out overlap controls.

prepare -> segment --scope real -> review real contact sheets -> baseline ->
infer -> segment --scope generated -> review generated sheets -> evaluate -> report.
Review records are written only by an explicit review command after visual inspection.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import crossview_geometry_eval as geom
import crossview_geometry_dense_gt as dense
from crossview_geometry_matched_eval import mask_distances, paired_deltas
from vggtdiff_white_car import BASE, REPO, digest, save
from vggtdiff_latest_coverage_ab import montage

OUT = BASE / 'vggtdiff_consistency_abc_real16_v1'
SEEDS = [20261026, 20261027, 20261028]
# First uninterrupted, fully visible real-video segment. References never enter GT.
TARGET_ORDINALS = list(range(5, 21))
REFERENCE_ORDINALS = [0, 2, 4, 21, 23, 24]
CHAIN_REFS = list(range(6))
CALLS = {
    'A0': dict(targets=list(range(8)), source='real'),
    'A1': dict(targets=list(range(8, 16)), source='real'),
    'B1': dict(targets=list(range(8, 16)), source='A0'),
    'C': dict(targets=list(range(4, 12)), source='real'),
    'OA': dict(targets=list(range(6, 14)), source='real'),
    'OB': dict(targets=list(range(6, 14)), source='A0'),
}
PAIR_IDS = [(i, j) for i in range(16) for j in range(i + 1, min(i + 3, 16))]
TRIPLES = [list(range(i, i + 3)) for i in range(14)]
BOUNDARY_PAIRS = [(6, 8), (7, 8), (7, 9)]
TRACK_FIELDS = ['track_id','n_views','conflict','heldout_frame','fit_a','fit_b','min_angle',
                'parallax_deg','reason','error_px','depth_a','depth_b','depth_held','triple_index']


def read(p):
    return json.loads(Path(p).read_text())


def frozen_write(p, value):
    value=json.loads(json.dumps(value))
    if p.exists() and read(p) != value:
        raise ValueError(f'Frozen file differs: {p}')
    save(p, value)


def verify_files(rows):
    for r in rows:
        if digest(Path(r['path'])) != r['sha256']:
            raise ValueError(f"Frozen file changed: {r['path']}")


def validate():
    m = read(OUT/'manifest.json')
    verify_files(m['frozen_files'])
    for r in m['weights']:
        s = Path(r['path']).stat()
        if (s.st_size, s.st_mtime_ns) != (r['size'], r['mtime_ns']):
            raise ValueError(f"Weight changed: {r['path']}")
    return read(OUT/'protocol.json')


def prepare():
    import vggtdiff_real_front_references as real
    bridge = real.bridge
    meta = read(bridge.ROOT/'frames_manifest.json')['images']
    targets = [meta[i]['video_frame_index'] for i in TARGET_ORDINALS]
    references = [meta[i]['video_frame_index'] for i in REFERENCE_ORDINALS]
    assert len(set(targets)) == 16 and not set(targets) & set(references)
    OUT.mkdir(parents=True, exist_ok=True)
    old = bridge.anchors.FRAMES
    try:
        bridge.anchors.FRAMES = targets + references
        rows = real.prepare_real(OUT)
    finally:
        bridge.anchors.FRAMES = old
    by_id = {r['original_frame']: r for r in rows}
    targets.sort(key=lambda f: by_id[f]['angle'])
    references.sort(key=lambda f: by_id[f]['angle'])
    def export(ids, role):
        result = []
        for i, f in enumerate(ids):
            r = by_id[f]
            assert r['valid_fraction'] > .99999
            assert r['distortion_rotation_projection_check_max_px'] < 1e-3
            assert Image.open(r['image']).size == (832, 480)
            result.append(dict(index=i, original_frame=i if role=='gt' else 100+i,
                video_frame=f, image=r['image'], sha256=r['sha256'],
                mask=str(OUT/'masks'/role/f'{i:02d}.png'),
                K=r['intrinsics'], T=r['w2c'], azimuth_deg=r['angle'],
                calibration=str(OUT/'real_calibration'/f'camera_{f}.json')))
        return result
    gt, refs = export(targets, 'gt'), export(references, 'references')
    gaps = np.diff([r['azimuth_deg'] for r in gt])
    assert np.all(gaps > 0) and np.max(gaps) < 15, gaps
    prior = read(BASE/'crossview_geometry_matched_gt_v1/protocol.json')
    protocol = dict(seeds=SEEDS, steps=50, resolution=[832,480], vram_limit_gib=20,
        checkpoint=prior['checkpoint'], omega=prior['omega'], base=prior['base'],
        calls=CALLS, shared_first_batch='A0 reused exactly by B, no cherry-picking',
        chain_reference_indices=CHAIN_REFS, chain_reference_rule='First six A0 outputs. Overlap cameras6,7 excluded from source conditions, so overlap is held out.',
        target_video_frames=targets, reference_video_frames=references,
        primary_pairs=PAIR_IDS, local_triples=TRIPLES, boundary_pairs=BOUNDARY_PAIRS,
        boundary_triples=[[6,7,8],[7,8,9]], overlap_indices=[6,7],
        matcher=dict(type='SIFT', nfeatures=12000, ratio=.75, mutual=True, epipolar_prefilter=False),
        support_minimum_matches=30, coverage_grid=[8,6], primary_parallax_deg=1.,
        camera_status='Registered SfM OPENCV cameras mapped by frozen SfM/Omega Sim3; estimated, not pose GT.',
        real_image_processing='Original video pixels; distortion correction and rotation at unchanged camera centers, with exact output K. No fitted generated-image alignment.',
        statistics='Fixed pairs/triples; per-pair generated-minus-real differences. Low support retained. Three seeds individually and descriptive range; no independent point-level significance.',
        regions=['vehicle','background_excluding_15px_dilated_vehicle'],
        continuation_policy='No automatic acceptance: compare all three layers with real baseline, support and coverage, then inspect masks and visuals. Missing evidence cannot pass. Unknown areas only local compatibility, not geometry accuracy.',
        quality_acceptance_thresholds=None)
    frozen_write(OUT/'targets.json', dict(frames=gt))
    frozen_write(OUT/'references.json', dict(frames=refs))
    frozen_write(OUT/'protocol.json', protocol)
    modules=[Path(__file__),Path(geom.__file__),Path(dense.__file__),
             REPO/'scripts/crossview_geometry_matched_eval.py',Path(real.__file__),Path(bridge.__file__)]
    files=modules+[OUT/'targets.json',OUT/'references.json',OUT/'protocol.json',
        bridge.OUT/'alignment.json',bridge.ROOT/'frames_manifest.json']
    files += [bridge.ROOT/'colmap/sparse/0'/n for n in ['images.bin','cameras.bin','points3D.bin']]
    files += [Path(r[k]) for r in rows for k in ['image','full_image']]
    files += [Path(r['calibration']) for r in gt+refs]
    weights=[]
    for p in [Path(prior['checkpoint']),Path(prior['omega'])]:
        s=p.stat();weights.append(dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns))
    files.append(Path(prior['checkpoint']).with_name('config.json'))
    frozen_write(OUT/'manifest.json',dict(frozen_files=[dict(path=str(p.resolve()),sha256=digest(p)) for p in dict.fromkeys(files)],weights=weights))
    for p in modules:
        dest=OUT/'implementation'/p.name;dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,dest)
    for name, frames in [('gt',gt),('references',refs)]:
        montage([Path(r['image']) for r in frames],
            [f"{r['index']:02d} | real{r['video_frame']} | {r['azimuth_deg']:.2f}deg" for r in frames],
            OUT/f'{name}_contact.jpg', columns=4)
    if not (OUT/'status.json').exists():save(OUT/'status.json',dict(state='prepared_pending_real_masks_and_visual_review'))
    print('PREPARED',OUT,targets,references,flush=True)


def preflight(device='cuda'):
    p=validate()
    import torch
    result=dict(time=time.time(),device=device,torch_version=torch.__version__,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        if device=='cuda':
            torch.ones(1,device='cuda');torch.cuda.synchronize()
        result['state']='passed'
    except Exception as e:
        result.update(state='blocked_cuda',error=str(e))
    save(OUT/'preflight.json',result)
    print(json.dumps(result),flush=True)
    if result['state']!='passed':raise RuntimeError(result['error'])
    return p


def select_mask(candidates):
    choices=[]
    for c in ([] if candidates is None else candidates.candidates):
        m=c.mask.cpu().numpy()>.5
        ys,xs=np.where(m)
        if not len(xs):continue
        fraction=m.mean();below=m[140:].sum()/m.sum()
        score=float(c.score)+2*fraction+below-.0008*np.linalg.norm([xs.mean()-416,ys.mean()-330])
        if .05<fraction<.7 and below>.8:choices.append((score,m))
    if not choices:raise RuntimeError('No eligible white-car mask; inspect image')
    return max(choices,key=lambda x:x[0])[1]


def generated_rows(seed,call):
    root=OUT/f'seed_{seed}'/call
    result=read(root/'result.json');verify_files(result['frames'])
    p=validate()
    trajectory=read(root/'trajectory.json')
    if digest(root/'trajectory.json')!=result['trajectory_sha256']:
        raise ValueError('Generation trajectory changed')
    verify_files(trajectory['source_images'])
    metas=read(OUT/'targets.json')['frames']
    for j, i in enumerate(p['calls'][call]['targets']):
        np.testing.assert_array_equal(trajectory['target_w2c'][j],metas[i]['T'])
        np.testing.assert_array_equal(trajectory['target_intrinsics'][j],metas[i]['K'])
    if result['protocol_sha256']!=digest(OUT/'protocol.json'):raise ValueError('Different generation protocol')
    metas=read(OUT/'targets.json')['frames']
    if [r['index'] for r in result['frames']]!=p['calls'][call]['targets']:raise ValueError('Wrong target ordering')
    return [dict(metas[r['index']],image=r['path'],sha256=r['sha256'],mask=str(root/'masks'/f"{r['index']:02d}.png")) for r in result['frames']]


def mask_manifest(scope,seed=None,call=None):
    return OUT/'real_masks.json' if scope=='real' else OUT/f'seed_{seed}'/call/'mask_stats.json'


@contextmanager
def cpu_unpinned_memory(device):
    """SAM3 pins temporary CPU metadata even for CPU inference; no-op only there."""
    import torch
    if device!='cpu':
        yield
        return
    original=torch.Tensor.pin_memory
    try:
        torch.Tensor.pin_memory=lambda tensor,*a,**k: tensor if tensor.device.type=='cpu' else original(tensor,*a,**k)
        yield
    finally:
        torch.Tensor.pin_memory=original


def segment(scope,seeds,calls,device):
    validate()
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    import torch
    torch.set_num_threads(8)
    if device=='cuda':preflight(device)
    model=SAM3Segmenter(SimpleNamespace(source_dir=str(REPO/'thirdparty/sam3'),
        checkpoint_path=str(REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
        device=device,confidence_threshold=.15,instance_strategy='best_score'))
    if device=='cpu':
        # SAM3 unconditionally preallocates its optional position cache on CUDA.
        # Disable only that cache while building this CPU model; forward computes
        # the identical analytic position encoding on the input device.
        sys.path.insert(0,str(REPO/'thirdparty/sam3'))
        import sam3.model_builder as builder
        from sam3.model.decoder import TransformerDecoder
        original=builder._create_position_encoding
        original_coords=TransformerDecoder._get_coords
        try:
            TransformerDecoder._get_coords=staticmethod(lambda H,W,device: original_coords(H,W,'cpu'))
            builder._create_position_encoding=lambda precompute_resolution=None: original(None)
            model._load_processor()
        finally:
            builder._create_position_encoding=original
            TransformerDecoder._get_coords=staticmethod(original_coords)
    groups=[(None,None,read(OUT/'targets.json')['frames']+read(OUT/'references.json')['frames'])] if scope=='real' else [(s,c,generated_rows(s,c)) for s in seeds for c in calls]
    for seed,call,frames in groups:
        rows=[];overlays=[]
        manifest=mask_manifest(scope,seed,call)
        prior_masks={r['path']:r for r in read(manifest)['frames']} if manifest.exists() else {}
        for r in frames:
            image=Path(r['image']);assert digest(image)==r['sha256']
            rgb=np.asarray(Image.open(image).convert('RGB'));dest=Path(r['mask']);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and str(dest) in prior_masks and prior_masks[str(dest)]['image_sha256']!=r['sha256']:
                raise ValueError(f'Stale mask for changed image: {dest}')
            if not dest.exists():
                with torch.inference_mode(), cpu_unpinned_memory(device):
                    candidates=model.segment_candidates(torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(device)/255,'white car')
                m=select_mask(candidates);Image.fromarray(m.astype('uint8')*255).save(dest)
            m=geom.read_mask(dest)>0
            assert m.shape==(480,832) and m.sum()>1000
            over=rgb.copy();over[m]=(.55*rgb[m]+.45*np.array([0,255,0])).astype('uint8')
            cv2.drawContours(over,cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,0,0),1)
            overlay=dest.with_name(dest.stem+'_overlay.jpg');Image.fromarray(over).save(overlay);overlays.append(overlay)
            rows.append(dict(path=str(dest),sha256=digest(dest),image_sha256=digest(image),area=int(m.sum()),
                touches_border=bool(m[:2].any() or m[-2:].any() or m[:,:2].any() or m[:,-2:].any())))
            print('MASK',scope,seed,call,r['original_frame'],rows[-1]['area'],flush=True)
        manifest=mask_manifest(scope,seed,call)
        frozen_write(manifest,dict(frames=rows))
        montage(overlays,[f"{r['original_frame']} | real {r['video_frame']}" for r in frames],manifest.with_name('mask_review.jpg'),columns=4)


def review(scope,seeds,calls,note):
    validate()
    if not note.strip():raise ValueError('Record concrete visual review observations')
    groups=[(None,None)] if scope=='real' else [(s,c) for s in seeds for c in calls]
    for s,c in groups:
        manifest=mask_manifest(scope,s,c);verify_files(read(manifest)['frames'])
        root=OUT if scope=='real' else OUT/f'seed_{s}'/c
        save(root/'visual_review.json',dict(state='accepted',note=note,mask_manifest_sha256=digest(manifest),time=time.time()))


def require_review(scope,seed=None,call=None):
    manifest=mask_manifest(scope,seed,call)
    verify_files(read(manifest)['frames'])
    root=OUT if scope=='real' else OUT/f'seed_{seed}'/call
    frames=(read(OUT/'targets.json')['frames']+read(OUT/'references.json')['frames']) if scope=='real' else generated_rows(seed,call)
    records=read(manifest)['frames']
    if len(frames)!=len(records) or any(r['image_sha256']!=f['sha256'] for r,f in zip(records,frames)):
        raise ValueError('Masks were produced for different image pixels')
    review=read(root/'visual_review.json')
    if review['state']!='accepted' or review['mask_manifest_sha256']!=digest(manifest):raise ValueError('Visual/mask review missing or stale')


def infer(seeds,calls):
    p=preflight();require_review('real')
    sys.path.insert(0,str(REPO/'thirdparty/VGGT-Diff'))
    from vggtdiff import VGGTDiff
    import torch
    model=None
    real=read(OUT/'references.json')['frames'];gt=read(OUT/'targets.json')['frames']
    for seed in seeds:
        for name in calls:
            spec=p['calls'][name];root=OUT/f'seed_{seed}'/name
            if (root/'result.json').exists():generated_rows(seed,name);continue
            if spec['source']=='real':refs=real
            else:
                require_review('generated',seed,'A0')
                pool={r['index']:r for r in generated_rows(seed,'A0')}
                refs=[pool[i] for i in p['chain_reference_indices']]
                assert not set(r['index'] for r in refs)&set(spec['targets'])
            if model is None:
                model=VGGTDiff(p['checkpoint'],p['omega'],base_model_dir=p['base'],device='cuda',
                    vram_limit_gib=p['vram_limit_gib'],training_mode='nvs',load_full_checkpoint_directly=True)
                assert not model.pipe.dit.training
            targets=[gt[i] for i in spec['targets']]
            trajectory=dict(source_w2c=[r['T'] for r in refs],source_intrinsics=[r['K'] for r in refs],
                target_w2c=[r['T'] for r in targets],target_intrinsics=[r['K'] for r in targets],
                target_indices=spec['targets'],source_images=[dict(path=r['image'],sha256=r['sha256']) for r in refs])
            frozen_write(root/'trajectory.json',trajectory)
            images=[Image.open(r['image']).convert('RGB') for r in refs]
            w=np.asarray([r['T'] for r in refs+targets],np.float32);k=np.asarray([r['K'] for r in refs+targets],np.float32)
            assert w.shape==(14,4,4) and k.shape==(14,3,3)
            save(root/'status.json',dict(state='generating',seed=seed,call=name))
            print('GENERATING',seed,name,flush=True);start=time.time();torch.cuda.reset_peak_memory_stats()
            result=model.generate(images,w,k,height=480,width=832,steps=p['steps'],seed=seed)
            assert len(result)==8
            frames=[]
            for r,img in zip(targets,result):
                assert img.size==(832,480)
                path=root/'raw'/f"{r['index']:02d}.png";path.parent.mkdir(parents=True,exist_ok=True);img.save(path)
                frames.append(dict(index=r['index'],path=str(path),sha256=digest(path)))
            montage([Path(r['path']) for r in frames],[f"{name} | target {r['index']}" for r in frames],root/'contact.jpg',columns=4)
            validate();verify_files(trajectory['source_images'])
            save(root/'result.json',dict(seed=seed,call=name,frames=frames,elapsed_seconds=time.time()-start,
                protocol_sha256=digest(OUT/'protocol.json'),trajectory_sha256=digest(root/'trajectory.json'),
                max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,dit_eval=not model.pipe.dit.training))
            save(root/'status.json',dict(state='generated_pending_masks_and_review'))


def load_features(rows,root,region):
    frames=[];(root/'features').mkdir(parents=True,exist_ok=True)
    for r in rows:
        assert digest(Path(r['image']))==r['sha256']
        image=geom.read_image(r['image']);mask=geom.read_mask(r['mask'])
        if region=='background':mask=255-cv2.dilate(mask,np.ones((31,31),np.uint8))
        signature=dict(image_sha256=r['sha256'],mask_sha256=digest(Path(r['mask'])),region=region,
                       protocol_sha256=digest(OUT/'protocol.json'))
        path=root/'features'/f"{r['original_frame']}.npz";sig=path.with_suffix('.json')
        if path.exists() and sig.exists() and read(sig)==signature:
            with np.load(path) as d:xy,des=d['xy'],d['descriptors']
        else:
            xy,des=geom.extract_features(image,mask);np.savez_compressed(path,xy=xy,descriptors=des);save(sig,signature)
        frames.append(dict(r,K=np.array(r['K']),T=np.array(r['T']),mask=mask,xy=xy,descriptors=des))
    return frames


def coverage_fraction(points,mask,grid=(8,6)):
    """Exact vectorized equivalent of the legacy mask-grid coverage definition."""
    ys,xs=np.nonzero(mask)
    if not len(xs):return None
    x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
    columns,rows=grid
    occupied_mask=np.unique(((ys-y0)*rows//(y1-y0))*columns+(xs-x0)*columns//(x1-x0))
    points=np.asarray(points)
    if not len(points):return 0.
    keep=(points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)
    q=points[keep]
    cells=((q[:,1]-y0)*rows/(y1-y0)).astype(int)*columns+((q[:,0]-x0)*columns/(x1-x0)).astype(int)
    return float(len(np.intersect1d(np.unique(cells),occupied_mask))/len(occupied_mask))


def pair_metrics(a,b):
    ia,ib=geom.mutual_ratio_matches(a['descriptors'],b['descriptors'])
    pa,pb=a['xy'][ia],b['xy'][ib]
    f,t=geom.fundamental_matrix(a['K'],a['T'],b['K'],b['T'])
    degenerate=bool(np.linalg.norm(t)<1e-6)
    row=dict(frame_a=a['original_frame'],frame_b=b['original_frame'],n_matches=len(ia),
        low_support=len(ia)<30,evidence='insufficient' if len(ia)<30 else 'ok',
        translation_norm=float(np.linalg.norm(t)),degenerate_translation=degenerate,
        coverage_a=coverage_fraction(pa,a['mask']),coverage_b=coverage_fraction(pb,b['mask']),
        azimuth_gap_deg=geom.angular_gap(a['azimuth_deg'],b['azimuth_deg']))
    row['azimuth_bucket']=geom.angle_bucket(row['azimuth_gap_deg'])
    errors=geom.sampson_sqrt_px(f,pa,pb) if len(ia) and not degenerate else np.zeros(0)
    row.update(geom.epi_summary(errors))
    row['reason']='degenerate_translation' if degenerate else ('ok' if len(ia) else 'no_matches')
    for radius in [2,4]:
        keep=geom.points_inside(pa,geom.erode_mask(a['mask'],radius)) & geom.points_inside(pb,geom.erode_mask(b['mask'],radius))
        row[f'n_matches_erode{radius}']=int(keep.sum())
        row[f'epi_median_erode{radius}_px']=geom.epi_summary(errors[keep] if len(errors) else [])['epi_median_px']
    return row,ia,ib


def summarize_pairs(rows):
    s=dense.pair_summary(rows)
    s['supported_fraction']=s['supported_pairs']/s['n_pairs'] if s['n_pairs'] else None
    return s


def assess(frames,root):
    root.mkdir(parents=True,exist_ok=True);byid={r['original_frame']:r for r in frames}
    ids=set(byid);fixed=[pair for pair in PAIR_IDS if set(pair)<=ids]
    rows=[];matches={}
    for a,b in fixed:
        row,ia,ib=pair_metrics(byid[a],byid[b]);rows.append(row);matches[(a,b)]=(ia,ib)
    geom.write_csv(root/'pairs.csv',rows,geom.PAIR_FIELDS)
    raw=[];closed=[];triple_rows=[]
    for ti,triple in enumerate(TRIPLES):
        if not set(triple)<=ids:continue
        sub={k:v for k,v in matches.items() if set(k)<=set(triple)}
        edges={k:set(zip(a.tolist(),b.tolist())) for k,(a,b) in sub.items()}
        local=[];cycle=[]
        for track in geom.build_tracks(sub):
            if track['n_views']<3:continue
            track=dict(track,track_id=ti*100000+track['track_id'])
            attempts=[]
            for angle in [.5,1.,2.]:attempts+=geom.heldout_attempts(track,byid,angle)
            for attempt in attempts:attempt['triple_index']=ti
            local+=attempts
            if dense.direct_cycle(track,edges):cycle+=attempts
        raw+=local;closed+=cycle
        triple_rows.append(dict(indices=triple,raw=dense.summarized_attempts(local),cycle=dense.summarized_attempts(cycle)))
    for name,attempts in [('raw',raw),('cycle',closed)]:geom.write_csv(root/f'tracks_{name}.csv',attempts,TRACK_FIELDS)
    boundary=lambda r:(r['frame_a'],r['frame_b']) in BOUNDARY_PAIRS
    boundary_track=lambda r:r['triple_index'] in [6,7]
    summary=dict(pairs=summarize_pairs(rows),within=summarize_pairs([r for r in rows if not boundary(r)]),
        boundary=summarize_pairs([r for r in rows if boundary(r)]),
        triples=triple_rows,raw=dense.summarized_attempts(raw),cycle=dense.summarized_attempts(closed),
        boundary_raw=dense.summarized_attempts([r for r in raw if boundary_track(r)]),
        boundary_cycle=dense.summarized_attempts([r for r in closed if boundary_track(r)]))
    save(root/'summary.json',summary)
    return rows,summary


def same_camera(a,b):
    np.testing.assert_array_equal(a['K'],b['K']);np.testing.assert_array_equal(a['T'],b['T'])
    ia,ib=geom.mutual_ratio_matches(a['descriptors'],b['descriptors'])
    distances=np.linalg.norm(a['xy'][ia]-b['xy'][ib],axis=1)
    iou,mean,p90=mask_distances(a['mask'],b['mask'])
    return dict(index=a['index'],n_matches=len(ia),low_support=len(ia)<30,
        displacement_median_px=float(np.median(distances)) if len(distances) else None,
        displacement_p90_px=float(np.percentile(distances,90)) if len(distances) else None,
        coverage_a=coverage_fraction(a['xy'][ia],a['mask']),coverage_b=coverage_fraction(b['xy'][ib],b['mask']),
        mask_iou=iou,boundary_mean_px=mean,boundary_p90_px=p90)


def reference_metrics(frames,refs,root):
    rows=[]
    for a in frames:
        for b in refs:
            r,_,_=pair_metrics(a,b)
            r['azimuth_gap_deg']=geom.angular_gap(a['azimuth_deg'],b['azimuth_deg']);rows.append(r)
    geom.write_csv(root/'references.csv',rows,geom.PAIR_FIELDS)
    return rows


def baseline():
    validate();require_review('real')
    summaries={}
    for region in ['vehicle','background']:
        root=OUT/'evaluation'/region/'gt'
        gt=load_features(read(OUT/'targets.json')['frames'],root,region)
        refs=load_features(read(OUT/'references.json')['frames'],root/'references',region)
        _,s=assess(gt,root);refrows=reference_metrics(gt,refs,root)
        s['reference_near']=summarize_pairs([r for r in refrows if r['azimuth_gap_deg']<=15])
        summaries[region]=s
    save(OUT/'baseline_summary.json',summaries)
    print('BASELINE_COMPLETE',flush=True)


def evaluate(seeds):
    validate();require_review('real')
    all_results=read(OUT/'comparison_summary.json') if (OUT/'comparison_summary.json').exists() else {}
    for region in ['vehicle','background']:
        regionroot=OUT/'evaluation'/region
        gt=load_features(read(OUT/'targets.json')['frames'],regionroot/'gt',region)
        refs=load_features(read(OUT/'references.json')['frames'],regionroot/'gt/references',region)
        gt_pairs,gt_summary=assess(gt,regionroot/'gt');gt_ref=reference_metrics(gt,refs,regionroot/'gt')
        region_results={'gt':gt_summary,'seeds':dict(all_results.get(region,{}).get('seeds',{}))}
        for seed in seeds:
            frames={}
            for call in CALLS:
                require_review('generated',seed,call)
                frames[call]=load_features(generated_rows(seed,call),regionroot/f'seed_{seed}'/call,region)
            result={}
            for name,seq in [('A',frames['A0']+frames['A1']),('B',frames['A0']+frames['B1']),('C',frames['C'])]:
                root=regionroot/f'seed_{seed}'/name
                rows,s=assess(seq,root);refrows=reference_metrics(seq,refs,root)
                gt_by_triple={tuple(t['indices']):t for t in gt_summary['triples']}
                triple_deltas=[]
                for triple in s['triples']:
                    for kind in ['raw','cycle']:
                        a=gt_by_triple[tuple(triple['indices'])][kind];b=triple[kind]
                        row=dict(indices=triple['indices'],kind=kind,gt_tracks=a['n_tracks'],generated_tracks=b['n_tracks'],
                            gt_failure_rate=a['computational_failure_rate'],generated_failure_rate=b['computational_failure_rate'],
                            evidence='insufficient' if min(a['n_tracks'],b['n_tracks'])<20 else 'measured_not_accepted')
                        for metric in ['computed_inside_image_median_px','computed_inside_image_p90_px']:
                            row['gt_'+metric]=a[metric];row['generated_'+metric]=b[metric]
                            row['delta_'+metric]=b[metric]-a[metric] if a[metric] is not None and b[metric] is not None else None
                        triple_deltas.append(row)
                geom.write_csv(root/'delta_triples_gt.csv',triple_deltas,list(triple_deltas[0]))
                fixed=[(r['frame_a'],r['frame_b']) for r in rows]
                deltas,paired=paired_deltas(gt_pairs,rows,fixed)
                geom.write_csv(root/'delta_gt.csv',deltas,list(deltas[0]))
                bd,bs=paired_deltas(gt_pairs,rows,BOUNDARY_PAIRS)
                geom.write_csv(root/'delta_boundary_gt.csv',bd,list(bd[0]))
                near=[(r['frame_a'],r['frame_b']) for r in refrows if r['azimuth_gap_deg']<=15]
                rd,rs=paired_deltas(gt_ref,refrows,near)
                if rd:geom.write_csv(root/'delta_references_gt.csv',rd,list(rd[0]))
                same=[same_camera(gt[r['index']],r) for r in seq]
                geom.write_csv(root/'same_camera_gt.csv',same,list(same[0]))
                result[name]=dict(geometry=s,paired_gt=paired,boundary_gt=bs,reference_gt=rs,same_camera_gt=same)
            # Exact same pairs and triples across C (joint), A (split), B (relay).
            for region_name,aux in [('overlap_real','OA'),('overlap_relay','OB')]:
                pool={r['index']:r for r in frames[aux]}
                same=[same_camera(frames['A0'][i],pool[i]) for i in [6,7]]
                result[region_name]=same
                root=regionroot/f'seed_{seed}';geom.write_csv(root/f'{region_name}.csv',same,list(same[0]))
            result['continuation_decision']='requires_three_layer_review; no automatic acceptance'
            save(regionroot/f'seed_{seed}/summary.json',result);region_results['seeds'][str(seed)]=result
        all_results[region]=region_results
    save(OUT/'comparison_summary.json',all_results)
    print('EVALUATION_COMPLETE',flush=True)


def seed_ranges(data):
    result={}
    for group in ['A','B','C']:
        values=[r[group]['geometry']['boundary']['supported_median_pair_epi_px'] for r in data['vehicle']['seeds'].values()]
        finite=[v for v in values if v is not None]
        result[group]=dict(n_seeds=len(values),n_measured=len(finite),min=min(finite) if finite else None,
            max=max(finite) if finite else None,median=float(np.median(finite)) if finite else None)
    return result


def report():
    validate()
    summary=OUT/'comparison_summary.json'
    available=summary.exists()
    complete=available and all(set(read(summary).get(r,{}).get('seeds',{}))==set(map(str,SEEDS)) for r in ['vehicle','background'])
    lines=['# 真实16视角三层一致性实验','',
        '相机为估计值；GT是真实视频像素。主匹配仅在白车内，背景独立报告。832×480；SIFT双向ratio0.75；不做极线预筛、不做图像配准。',
        '', '| 调用 | 目标索引 | 参考 |', '|---|---|---|']
    for name,c in CALLS.items():lines.append(f"| {name} | {c['targets']} | {c['source']} |")
    lines+=['','A=A0+A1；B=A0+B1；C=4～11。A/B共用完全相同的第一批。三组比较相同边界(6,8)、(7,8)、(7,9)和三元组(6,7,8)、(7,8,9)。',
        'OA/OB是额外的两相机重叠对照，目标6～13，重叠6、7没有作为OB参考；B1和OB使用相同A0前六张参考，避免将输入复现误认为衔接。',
        '三种子固定20261026/27/28；每个种子六次八目标调用，共144张生成图，含重复相机。参考与GT互斥。',
        '', '## 数据与审阅状态','']
    targets=read(OUT/'targets.json')['frames'];refs=read(OUT/'references.json')['frames']
    lines += [f"GT视频帧：{[r['video_frame'] for r in targets]}",f"真实参考帧：{[r['video_frame'] for r in refs]}",
        f"GT方位角：{targets[0]['azimuth_deg']:.3f}°～{targets[-1]['azimuth_deg']:.3f}°。距离变化保留，不将实拍相机改成等半径。",'']
    if (OUT/'baseline_summary.json').exists():
        lines+=['## 真实基线（与生成结果独立）','', '| 区域 | 支持/固定配对 | 极线med/P90 | 三视角med/P90 | 轨迹数/失败率 |', '|---|---|---|---|---|']
        for region,g in read(OUT/'baseline_summary.json').items():
            a=g['pairs'];t=g['raw']
            lines.append(f"| {region} | {a['supported_pairs']}/{a['n_pairs']} | {a['supported_median_pair_epi_px']}/{a['supported_median_pair_p90_px']} | {t['computed_inside_image_median_px']}/{t['computed_inside_image_p90_px']} | {t['n_tracks']}/{t['computational_failure_rate']} |")
        lines+=['','误差单位px；三视角主结果为原始局部轨迹。背景独立统计，不能替代白车证据。','']
    if not available:
        lines+=['实验尚未完成，不能得出生成一致性结论。','']
        if (OUT/'preflight.json').exists():lines+=['设备预检：```json',json.dumps(read(OUT/'preflight.json'),ensure_ascii=False,indent=2),'```','']
        lines+=['已准备/待运行项请参见status.json及README.md。缺少掩码、视觉审阅或生成图时，不输出虚构指标。']
    else:
        data=read(summary)
        if not complete:lines+=['当前只有部分种子结果，三种子实验未完成。','']
        save(OUT/'seed_ranges.json',seed_ranges(data))
        for region in ['vehicle','background']:
            lines += [f'## {region}：同一边界与批内','',
                '| seed/组 | 批内支持/总数 | 批内极线med/P90 | 边界支持/总数 | 边界极线med/P90 | 边界三视角med/P90 | 三视角轨迹数/失败率 |',
                '|---|---|---|---|---|---|---|']
            groups=[('GT',data[region]['gt'])]
            groups += [(f'{s}/{name}',r[name]['geometry']) for s,r in data[region]['seeds'].items() for name in ['A','B','C']]
            for label,g in groups:
                w,b,t=g['within'],g['boundary'],g['boundary_raw']
                lines.append(f"| {label} | {w['supported_pairs']}/{w['n_pairs']} | {w['supported_median_pair_epi_px']}/{w['supported_median_pair_p90_px']} | {b['supported_pairs']}/{b['n_pairs']} | {b['supported_median_pair_epi_px']}/{b['supported_median_pair_p90_px']} | {t['computed_inside_image_median_px']}/{t['computed_inside_image_p90_px']} | {t['n_tracks']}/{t['computational_failure_rate']} |")
            lines+=['','各三元组的原始/描述子闭环结果、视差0.5/1/2°、失败原因保存在tracks_*.csv和summary.json。原始结果为主，闭环为补充。','',
                '| seed/组 | 同GT支持/固定配对 | GT支持丢失 | 逐对极线增量 | 真实参考近邻增量 | 同相机偏移med | 轮廓IoU med |','|---|---|---|---|---|---|---|']
            for seed,r in data[region]['seeds'].items():
                for name in ['A','B','C']:
                    a=r[name];v=a['paired_gt'];same=a['same_camera_gt']
                    def med(k):
                        nums=[s[k] for s in same if s[k] is not None and not s['low_support']]
                        return float(np.median(nums)) if nums else None
                    lines.append(f"| {seed}/{name} | {v['joint_supported']}/{v['n_fixed']} | {v['lost_gt_support']} | {v['joint_median_paired_delta_px']} | {a['reference_gt']['joint_median_paired_delta_px']} | {med('displacement_median_px')} | {float(np.median([s['mask_iou'] for s in same]))} |")
            lines+=['','同相机与重叠相机每视角的med/P90、匹配数、双侧覆盖率、轮廓IoU和边界距离均保留CSV。背景轮廓统计仅表示车辆排除区差异，不作为背景结构主指标。',
                '', '| seed/重叠组 | 相机 | 匹配数 | 偏移med/P90 | IoU | 双侧覆盖 |','|---|---|---|---|---|---|']
            for seed,r in data[region]['seeds'].items():
                for name in ['overlap_real','overlap_relay']:
                    for v in r[name]:lines.append(f"| {seed}/{name} | {v['index']} | {v['n_matches']} | {v['displacement_median_px']}/{v['displacement_p90_px']} | {v['mask_iou']} | {v['coverage_a']}/{v['coverage_b']} |")
        lines+=['','## 三种子边界极线波动（白车）','', '```json',json.dumps(seed_ranges(data),indent=2),'```',
            '', '## 判断边界','',
            '不能仅凭低误差允许接力。需要同时核查相对真实基线的三层指标、支持量和空间覆盖、同相机GT偏差及重叠跳变。未预设通用像素合格线，本报告不自动宣称可继续扩展。',
            '匹配<30为证据不足；低支持配对保留，支持集合统计与全固定集合分开。三视角计算成功不等于几何正确。三个种子和重叠三元组不作为独立样本。',
            '未知区域没有真实GT，只能评价局部兼容性和漂移风险，不能验证真实几何准确性。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'status.json',dict(state='evaluated_pending_interpretation' if complete else 'implementation_ready_experiment_incomplete',
        completed_comparison=complete,real_baseline_completed=(OUT/'baseline_summary.json').exists(),preflight=read(OUT/'preflight.json') if (OUT/'preflight.json').exists() else None))


def main():
    global OUT
    cv2.setNumThreads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','segment','review','baseline','infer','evaluate','report'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--calls',nargs='+',choices=list(CALLS),default=list(CALLS))
    parser.add_argument('--scope',choices=['real','generated'],default='real')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--note',default='')
    args=parser.parse_args();OUT=args.output.resolve()
    if not set(args.seeds)<=set(SEEDS):parser.error('Seeds must come from frozen protocol')
    if args.action=='prepare':prepare()
    elif args.action=='preflight':preflight(args.device)
    elif args.action=='segment':segment(args.scope,args.seeds,args.calls,args.device)
    elif args.action=='review':review(args.scope,args.seeds,args.calls,args.note)
    elif args.action=='baseline':baseline()
    elif args.action=='infer':infer(args.seeds,args.calls)
    elif args.action=='evaluate':evaluate(args.seeds)
    else:report()


if __name__=='__main__':main()
