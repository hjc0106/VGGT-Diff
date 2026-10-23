"""Compare old/new batch7 between unchanged batches6/8 using vehicle geometry."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import itertools
import cv2
import numpy as np
from PIL import Image
import vggtdiff_batch06_08_to_batch07_continuous as job
import analyze_batch08_real_anchor_results as core

OUT=job.OUT/'evaluation_bridge_v1'
E67=job.chain.BASE/'crossview_geometry_batch06_batch07_continuous_v1'
E78=job.chain.BASE/'crossview_geometry_batch07_batch08_real6_v1'


def group(a,b):
    return f"within{a['batch']}" if a['batch']==b['batch'] else f"cross{a['batch']}{b['batch']}"


def prepare():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    torch.set_num_threads(4);cv2.setNumThreads(4)
    OUT.mkdir(exist_ok=True)
    job.chain.validate(job.OUT);job.chain.validate(job.B8);job.chain.validate(job.OLD7)
    meta67=core.read(E67/'frames.json')['frames'];meta78=core.read(E78/'frames.json')['frames']
    new_result=core.read(job.OUT/'result.json');new_t=core.read(job.OUT/'trajectory.json')
    assert new_result['cameras_sha256']==job.chain.digest(job.OUT/'trajectory.json')
    old_t=core.read(job.OLD7/'trajectory.json')
    for k in ['target_w2c','target_intrinsics','target_angles_deg']:np.testing.assert_array_equal(new_t[k],old_t[k])
    protocol=dict(image_size=[832,480],variants=['old7','new7'],image_count_per_variant=24,
        sources=dict(batch6=str(job.B6),old7=str(job.OLD7),new7=str(job.OUT),batch8=str(job.B8)),
        matcher='SIFT12000, mutual Lowe.75; vehicle-only reviewed masks; no epipolar/RANSAC prefilter, camera refit or error selection',
        support_minimum_matches=30,primary='All<=15deg pairs, separate within6/within7/within8/cross67/cross78/cross68',
        triples='22 consecutive local triples: six within each batch, two each left/right boundary. Raw and direct three-edge descriptor cycle.',
        minimum_parallax_degrees=[.5,1,2],primary_parallax_degrees=1,acceptance_threshold=None,
        interpretation='Compatibility with specified estimated-world cameras; not independent actual-pose GT. New7 boundary images include conditioning references.')
    core.save(OUT/'protocol.json',protocol)
    rows=[];segmenter=None;overlays=[]
    for variant in ['old7','new7']:
        for ident in range(40,64):
            batch=6 if ident<48 else (7 if ident<56 else 8)
            if batch==7 and variant=='new7':
                i=ident-48;r=new_result['frames'][i];image=Path(r['path'])
                mask_path=OUT/'masks/new7'/f'{ident}.png';mask_path.parent.mkdir(parents=True,exist_ok=True)
                if not mask_path.exists():
                    if segmenter is None:
                        segmenter=SAM3Segmenter(SimpleNamespace(source_dir=str(job.chain.REPO/'thirdparty/sam3'),checkpoint_path=str(job.chain.REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
                    rgb=np.asarray(Image.open(image).convert('RGB'));mask=core.choose_vehicle(segmenter,rgb)
                    Image.fromarray(mask.astype('uint8')*255).save(mask_path)
                row=dict(image=str(image),sha256=r['sha256'],K=new_t['target_intrinsics'][i],T=new_t['target_w2c'][i],azimuth_deg=r['angle'])
            else:
                if batch==6:old=next(f for f in meta67 if f['batch']==6 and f['local_index']==ident-40)
                else:old=next(f for f in meta78 if f['index']==ident)
                assert job.chain.digest(Path(old['image']))==old['sha256']
                assert job.chain.digest(Path(old['mask']))==old['mask_sha256']
                mask_path=OUT/'masks'/('old7' if batch==7 else f'batch{batch}')/f'{ident}.png'
                job.chain.frozen_copy(Path(old['mask']),mask_path)
                row={k:old[k] for k in ['image','sha256','K','T','azimuth_deg']}
            assert job.chain.digest(Path(row['image']))==row['sha256']
            assert Image.open(row['image']).size==(832,480)
            mask=core.geom.read_mask(mask_path)>0;assert mask.sum()>1000
            row.update(index=ident,original_frame=ident,batch=batch,variant=variant,mask=str(mask_path),mask_sha256=job.chain.digest(mask_path),mask_area=int(mask.sum()))
            rows.append(row)
            if batch==7 and variant=='new7':
                rgb=np.asarray(Image.open(row['image']).convert('RGB'));rgb=rgb.copy();rgb[mask]=(.55*rgb[mask]+.45*np.array([0,255,0])).astype('uint8')
                p=mask_path.with_name(f'{ident}_overlay.jpg');Image.fromarray(rgb).save(p);overlays.append(p)
            print('MASK',variant,ident,row['mask_area'],flush=True)
    del segmenter;torch.cuda.empty_cache()
    job.chain.montage(overlays,[f'New7 | {i} | {i*5.625:.3f} deg' for i in range(48,56)],OUT/'mask_review_new7.jpg',4,(832,480))
    core.save(OUT/'frames.json',dict(frames=rows))
    files=[Path(__file__).resolve(),Path(core.__file__).resolve(),Path(core.geom.__file__).resolve(),Path(core.dense.__file__).resolve(),Path(core.matched.__file__).resolve(),OUT/'protocol.json',OUT/'frames.json',E67/'summary.json',E78/'summary.json']
    files += [p/name for p in [job.OUT,job.OLD7,job.B8] for name in ['manifest.json','trajectory.json','result.json']]
    files += [job.B6/'result.json',job.B6.parent/'trajectory.json']
    files += list(dict.fromkeys(Path(r[k]) for r in rows for k in ['image','mask']))
    core.save(OUT/'manifest.json',dict(files=[dict(path=str(p),sha256=job.chain.digest(p)) for p in files]))
    for p in [Path(__file__).resolve(),Path(core.__file__).resolve()]:job.chain.frozen_copy(p,OUT/'implementation'/p.name)


def validate():
    for r in core.read(OUT/'manifest.json')['files']:assert job.chain.digest(Path(r['path']))==r['sha256']


def evaluate():
    validate();assert core.read(OUT/'mask_review.json')['state']=='accepted';cv2.setNumThreads(4)
    metadata=core.read(OUT/'frames.json')['frames'];summaries={}
    for variant in ['old7','new7']:
        root=OUT/variant;frames=[core.matched.feature_frame(r,root/'features') for r in metadata if r['variant']==variant]
        byid={f['index']:f for f in frames};matches=core.matched.build_matches(frames,root)
        rows=[]
        for (a,b),(ia,ib) in matches.items():
            fa,fb=byid[a],byid[b]
            r=core.geom.pair_record(a,b,fa['xy'][ia],fb['xy'][ib],fa['mask'],fb['mask'],fa['K'],fa['T'],fb['K'],fb['T'])
            r.update(azimuth_gap_deg=core.geom.angular_gap(fa['azimuth_deg'],fb['azimuth_deg']),group=group(fa,fb),index_gap=b-a)
            rows.append(r)
        core.geom.write_csv(root/'pairs.csv',rows,core.geom.PAIR_FIELDS+['group','index_gap'])
        # Save only local/boundary overlays; metrics retain all276 pairs.
        overlay_rows=[r for r in rows if r['group'] in ['cross67','cross78'] and r['azimuth_gap_deg']<=15 or (r['group']=='within7' and r['index_gap']==1)]
        core.geom._write_overlays(root,frames,matches,overlay_rows)
        pair_groups={}
        for name in ['within6','within7','within8','cross67','cross78','cross68']:
            sub=[r for r in rows if r['group']==name]
            pair_groups[name]=dict(all=core.dense.pair_summary(sub),primary15=core.dense.pair_summary([r for r in sub if r['azimuth_gap_deg']<=15]))
        groups={g:dict(raw=[],cycle=[]) for g in ['within6','left_boundary','within7','right_boundary','within8']}
        edges={k:set(zip(v[0].tolist(),v[1].tolist())) for k,v in matches.items()}
        working={i:dict(K=f['K'],T=f['T'],xy=f['xy'],mask=f['mask']) for i,f in byid.items()};triples=[]
        for start in range(40,62):
            ids=list(range(start,start+3));name='within6' if start<=45 else ('left_boundary' if start<=47 else ('within7' if start<=53 else ('right_boundary' if start<=55 else 'within8')))
            sub={k:v for k,v in matches.items() if set(k).issubset(ids)};raw=[];cycle=[]
            for track in core.geom.build_tracks(sub):
                if track['n_views']<3:continue
                track=dict(track,track_id=start*100000+track['track_id'])
                attempts=[a for threshold in [.5,1,2] for a in core.geom.heldout_attempts(track,working,threshold)]
                raw+=attempts
                if core.dense.direct_cycle(track,edges):cycle+=attempts
            groups[name]['raw']+=raw;groups[name]['cycle']+=cycle
            triples.append(dict(frames=ids,group=name,raw=core.dense.summarized_attempts(raw),cycle=core.dense.summarized_attempts(cycle)))
        tracks={name:{method:core.dense.summarized_attempts(v) for method,v in methods.items()} for name,methods in groups.items()}
        for name,methods in groups.items():
            for method,values in methods.items():core.geom.write_csv(root/f'tracks_{name}_{method}.csv',values,core.matched.TRACK_FIELDS)
        summary=dict(pair_groups=pair_groups,track_groups=tracks,triples=triples,
            primary_boundary_pairs=[{k:v for k,v in r.items() if k!='errors'} for r in rows if r['group'] in ['cross67','cross78'] and r['azimuth_gap_deg']<=15],
            adjacent7=[{k:v for k,v in r.items() if k!='errors'} for r in rows if r['group']=='within7' and r['index_gap']==1])
        core.save(root/'summary.json',summary);summaries[variant]=summary
        print('SUMMARY',variant,pair_groups,tracks,flush=True)
    # Verify the untouched neighbor batches and the old bridge reproduce prior audits.
    old67=core.read(E67/'summary.json');old78=core.read(E78/'summary.json')
    for variant,s in summaries.items():
        for current,expected in [('within6',old67['pair_groups']['batch6']),('within8',old78['pair_groups']['within8'])]:
            np.testing.assert_allclose(s['pair_groups'][current]['primary15']['supported_median_pair_epi_px'],expected['primary15']['supported_median_pair_epi_px'],atol=1e-10)
    for current,expected in [('within7',old78['track_groups']['within7']),('left_boundary',old67['track_groups']['boundary']),('right_boundary',old78['track_groups']['boundary'])]:
        for metric in ['computed_inside_image_median_px','computed_inside_image_p90_px']:
            np.testing.assert_allclose(summaries['old7']['track_groups'][current]['cycle'][metric],expected['cycle'][metric],atol=1e-10)
    core.save(OUT/'summary.json',summaries);validate()
    core.save(OUT/'audit.json',dict(state='passed',frozen_files_verified=True,variants=2,images_per_variant=24,pairs_per_variant=276,local_triples_per_variant=22,old_metrics_exactly_reproduced=True,quality_acceptance_threshold=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evaluate']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else evaluate()
