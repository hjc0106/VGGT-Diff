"""Three-way batch8 vehicle geometry, reusing the unchanged prior evaluator."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import cv2
import numpy as np
from PIL import Image
import analyze_batch08_real_anchor_results as core
import vggtdiff_batch08_real6_continuous as job

PREVIOUS = core.OUT
OUT = job.OUT / 'evaluation_threeway_v1'
RUNS = dict(core.RUNS, real6=job.OUT)


def configure():
    core.OUT = OUT
    core.RUNS = RUNS


def prepare():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    for row in core.read(PREVIOUS/'manifest.json')['files']:
        assert job.chain.digest(Path(row['path']))==row['sha256']
    previous = core.read(PREVIOUS/'frames.json')['frames']
    input_manifest = core.read(job.OUT/'input_geometry_check/manifest.json')
    for row in input_manifest['files']:
        assert job.chain.digest(Path(row['path']))==row['sha256']
    old_target = core.read(job.mixed.BASELINE/'trajectory.json')
    for variant, root in RUNS.items():
        _, t = job.chain.validate(root)
        result = core.read(root/'result.json')
        assert result['cameras_sha256']==job.chain.digest(root/'trajectory.json')
        for k in ['target_w2c','target_intrinsics','target_angles_deg']:
            np.testing.assert_array_equal(t[k],old_target[k])
        assert result['seed']==20261030 and result['steps']==50
        assert len(result['frames'])==8
    OUT.mkdir(exist_ok=True)
    protocol=core.read(PREVIOUS/'protocol.json')
    protocol.update(comparison='Three completed single-seed batch8 runs: six generated, four real+two generated, six real',
        boundary_group='Same prior batch7 images54/55 for three cross-boundary pairs<=15deg per run. These are conditioning images only in the first two runs; diagnostic compatibility only for pure-real control.',
        extra_reference_check='All outputs against the identical six real reference set, vehicle-only unfiltered epipolar matching; real6 conditions on all six, mixed on last four. Descriptive input compatibility, not independent GT accuracy.',
        historical_reproduction='Old two runs reuse exact reviewed masks and must reproduce primary pair and cycle-triple median/P90 metrics.',
        real6_frames=job.FRAMES)
    core.save(OUT/'protocol.json',protocol)
    frames=[];overlays=[];labels=[];segmenter=None
    target=core.read(job.OUT/'trajectory.json');result=core.read(job.OUT/'result.json')
    for variant in RUNS:
        for ident in range(54,64):
            mask_path=OUT/variant/'masks'/f'{ident}.png'
            mask_path.parent.mkdir(parents=True,exist_ok=True)
            if variant!='real6' or ident<56:
                original=next(r for r in previous if r['variant']==(variant if variant!='real6' else 'all_generated') and r['index']==ident)
                row=dict(original,variant=variant,mask=str(mask_path))
                job.chain.frozen_copy(Path(original['mask']),mask_path)
            else:
                i=ident-56;r=result['frames'][i];image=Path(r['path'])
                assert job.chain.digest(image)==r['sha256']
                rgb=np.asarray(Image.open(image).convert('RGB'))
                if not mask_path.exists():
                    if segmenter is None:
                        segmenter=SAM3Segmenter(SimpleNamespace(source_dir=str(job.chain.REPO/'thirdparty/sam3'),
                            checkpoint_path=str(job.chain.REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
                            device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
                    mask=core.choose_vehicle(segmenter,rgb)
                    Image.fromarray(mask.astype('uint8')*255).save(mask_path)
                row=dict(index=ident,original_frame=ident,variant=variant,image=str(image),mask=str(mask_path),
                    sha256=r['sha256'],K=target['target_intrinsics'][i],T=target['target_w2c'][i],azimuth_deg=r['angle'])
            assert job.chain.digest(Path(row['image']))==row['sha256']
            mask=core.geom.read_mask(mask_path)>0;ys,xs=np.where(mask)
            assert len(xs)>1000
            row.update(mask_sha256=job.chain.digest(mask_path),mask_area=int(mask.sum()),
                mask_bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)])
            frames.append(row)
            if variant=='real6' and ident>=56:
                rgb=np.asarray(Image.open(row['image']).convert('RGB'));over=rgb.copy()
                over[mask]=(.55*rgb[mask]+.45*np.array([0,255,0])).astype('uint8')
                dest=mask_path.with_name(f'{ident}_overlay.jpg');Image.fromarray(over).save(dest)
                overlays.append(dest);labels.append(f"real6 | {ident} | {row['azimuth_deg']:.3f} deg")
            print('MASK',variant,ident,row['mask_area'],flush=True)
    del segmenter
    torch.cuda.empty_cache()
    job.chain.montage(overlays,labels,OUT/'mask_review_real6.jpg',4,(832,480))
    core.save(OUT/'frames.json',dict(frames=frames))
    files=[Path(__file__).resolve(),Path(core.__file__).resolve(),Path(core.geom.__file__).resolve(),
        Path(core.dense.__file__).resolve(),Path(core.matched.__file__).resolve(),OUT/'frames.json',OUT/'protocol.json',
        PREVIOUS/'manifest.json',PREVIOUS/'summary.json',job.OUT/'evaluation_plan.json',job.OUT/'input_geometry_check/manifest.json']
    files += [root/name for root in RUNS.values() for name in ['manifest.json','trajectory.json','result.json']]
    files += [Path(r['path']) for r in input_manifest['files']]
    files += list((job.OUT/'input_geometry_check/features').glob('*.npz'))
    files += list(dict.fromkeys(Path(r[k]) for r in frames for k in ['image','mask']))
    core.save(OUT/'manifest.json',dict(files=[dict(path=str(p),sha256=job.chain.digest(p)) for p in files]))
    for p in [Path(__file__).resolve(),Path(core.__file__).resolve()]:
        job.chain.frozen_copy(p,OUT/'implementation'/p.name)
    core.save(OUT/'status.json',dict(state='prepared_pending_mask_review'))


def evaluate():
    configure()
    core.evaluate()
    current=core.read(OUT/'summary.json');previous=core.read(PREVIOUS/'summary.json')
    for variant in ['all_generated','real4_gen2']:
        for group in ['within_batch8','boundary']:
            np.testing.assert_allclose(current[variant]['pair_groups'][group]['primary15']['supported_median_pair_epi_px'],
                previous[variant]['pair_groups'][group]['primary15']['supported_median_pair_epi_px'],atol=1e-10)
            for k in ['computed_inside_image_median_px','computed_inside_image_p90_px']:
                np.testing.assert_allclose(current[variant]['track_groups'][group]['cycle'][k],previous[variant]['track_groups'][group]['cycle'][k],atol=1e-10)
    core.save(OUT/'historical_reproduction.json',dict(state='passed',scope='Both prior runs primary15 pair median and local cycle-track median/P90 reproduced within1e-10 for both within-batch and boundary groups.'))
    real_meta=core.read(job.OUT/'input_geometry_check/frames.json')['frames']
    real=[core.matched.feature_frame(r,OUT/'real_reference_features',cached=job.OUT/'input_geometry_check/features'/f"{r['original_frame']}.npz") for r in real_meta]
    extra={}
    for variant in RUNS:
        root=OUT/variant
        frames=[core.matched.feature_frame(r,root/'features',cached=root/'features'/f"{r['original_frame']}.npz") for r in core.read(OUT/'frames.json')['frames'] if r['variant']==variant and r['index']>=56]
        rows=[]
        for a in real:
            for b in frames:
                ia,ib=core.geom.mutual_ratio_matches(a['descriptors'],b['descriptors'])
                r=core.geom.pair_record(a['index'],b['index'],a['xy'][ia],b['xy'][ib],a['mask'],b['mask'],a['K'],a['T'],b['K'],b['T'])
                r.update(azimuth_gap_deg=core.geom.angular_gap(a['azimuth_deg'],b['azimuth_deg']),
                    real_video_frame=a['video_frame'],real_image_is_conditioning=(variant=='real6' or (variant=='real4_gen2' and a['index']>=2)))
                rows.append(r)
        core.geom.write_csv(root/'real_reference_pairs.csv',rows,core.geom.PAIR_FIELDS+['real_video_frame','real_image_is_conditioning'])
        extra[variant]=dict(scope='Same six real images for all runs; some/all are conditioning images. Not independent GT.',
            primary15=core.dense.pair_summary([r for r in rows if r['azimuth_gap_deg']<=15]),all=core.dense.pair_summary(rows))
    core.save(OUT/'real_reference_summary.json',extra)
    core.validate()
    print('THREEWAY_DONE',OUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate'])
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else evaluate()
