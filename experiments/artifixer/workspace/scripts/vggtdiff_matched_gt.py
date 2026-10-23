"""Frozen three-seed, eight-target NVS experiment at held-out real cameras."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'thirdparty/VGGT-Diff'), str(REPO), str(REPO / 'worldact')]
from vggtdiff_white_car import BASE, digest, save
from vggtdiff_latest_coverage_ab import montage

GT = BASE / 'vggtdiff_real_gt_batch06_targets_v1'
BASELINE = BASE / 'crossview_geometry_dense_real8_v1'
OUT = BASE / 'crossview_geometry_matched_gt_v1'
SEEDS = [20261026, 20261027, 20261028]


def read(path):
    return json.loads(path.read_text())


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def verify_rows(rows):
    for row in rows:
        if digest(absolute(row['path'])) != row['sha256']:
            raise ValueError(f"Frozen input changed: {row['path']}")


def validate():
    manifest = read(OUT / 'manifest.json')
    verify_rows(manifest['frozen_files'])
    for row in manifest['weights']:
        st = Path(row['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (row['size'], row['mtime_ns']):
            raise ValueError(f"Model weight changed: {row['path']}")


def prepare():
    verify_rows(read(GT / 'manifest.json')['frozen_files'])
    verify_rows(read(BASELINE / 'manifest.json')['frozen_files'])
    trajectory = read(GT / 'trajectory.json')
    gt_frames = read(BASELINE / 'frames.json')['frames']
    assert [r['original_frame'] for r in gt_frames] == trajectory['target_video_frames']
    real8 = read(BASE / 'crossview_geometry_real_validation_v1/manifest.json')['frames']
    references = sorted([r for r in real8 if r['role'] == 'reference'], key=lambda r:r['original_frame'])
    source_numbers = [0, 480, 720, 960, 1200, 1440]
    assert [r['original_frame'] for r in references] == source_numbers
    old_cameras = read(BASE / 'crossview_geometry_real_validation_v1/cameras.json')['sets']['legacy']['frames']
    OUT.mkdir(parents=True, exist_ok=True)
    files = [GT/'trajectory.json', GT/'cameras.json', GT/'manifest.json', BASELINE/'frames.json',
             BASELINE/'summary.json', BASELINE/'manifest.json', BASELINE/'protocol.json',
             REPO/'scripts/crossview_geometry_eval.py', REPO/'scripts/crossview_geometry_dense_gt.py', Path(__file__).resolve()]
    for i, r in enumerate(references):
        image = GT / 'source_views' / f'{i:02d}.png'
        assert digest(image) == r['sha256']
        assert np.allclose(trajectory['source_w2c'][i], old_cameras[str(r['original_frame'])]['T'])
        assert np.allclose(trajectory['source_intrinsics'][i], old_cameras[str(r['original_frame'])]['K'])
        r['image'], r['mask'] = str(image), str(absolute(r['mask']))
        assert digest(Path(r['mask'])) == r['mask_sha256']
        r['K'], r['T'] = trajectory['source_intrinsics'][i], trajectory['source_w2c'][i]
        files += [image, Path(r['mask'])]
    for r in gt_frames:
        files += [absolute(r['image']), absolute(r['mask']),
                  BASELINE/'features'/f"{r['original_frame']}.npz"]
    files += list((BASELINE/'matches').glob('*.npz'))
    prior = read(BASE / 'vggtdiff_real6_orbit_batch06_v1/manifest.json')
    model_config = Path(prior['checkpoint']).with_name('config.json')
    assert read(model_config)['model_family'] == 'nvs'
    files += [model_config]
    ids = trajectory['target_video_frames']
    pair_spec = read(BASELINE / 'pairs_spec.json')['pairs']
    primary = [[r['frame_a'], r['frame_b']] for r in pair_spec if r['azimuth_gap_deg'] <= 15]
    adjacent = [[ids[i],ids[i+1]] for i in range(7)]
    assert len(primary)==14
    protocol = dict(seeds=SEEDS, steps=50, resolution=[832,480], nvs_targets_per_call=8,
        checkpoint=prior['checkpoint'], omega=prior['omega'], base=prior['base'], vram_limit_gib=20,
        source_frames=source_numbers, target_frames=ids, excluded_from_sensitivity_subset=[1340,1300],
        primary_pairs=primary, adjacent_pairs=adjacent, local_triples=[ids[i:i+3] for i in range(6)],
        primary_pair_rule='All fourteen GT pairs with azimuth gap <=15 degrees, fixed before generation.',
        reference_pair_rule='All target/reference pairs reported; <=15 degrees primary; insufficient support retained.',
        matcher=dict(type='SIFT',nfeatures=12000,lowe_ratio=.75,mutual=True,epipolar_prefilter=False),
        support_minimum_matches=30, spatial_coverage_grid=[8,6],
        track_rule='Raw local triples plus direct descriptor-cycle subset. Never filter by heldout error.',
        same_view_rule='Descriptor coordinate displacement, mask IoU, symmetric contour distance. No fitted alignment.',
        statistics='Per-pair generated-minus-GT deltas on fixed pair IDs; joint-support subset separately. Lost support is not zero error. Three seeds reported individually; no point-level significance test.',
        output_scope='Same-camera real GT vs generated comparison; no Gaussian reconstruction or camera refit.',
        quality_acceptance_thresholds=None)
    save(OUT/'references.json',dict(frames=references))
    save(OUT/'targets.json',dict(frames=gt_frames))
    save(OUT/'trajectory.json',trajectory)
    save(OUT/'protocol.json',protocol)
    files += [OUT/'references.json', OUT/'targets.json', OUT/'trajectory.json', OUT/'protocol.json']
    weights=[]
    for p in [Path(prior['checkpoint']),Path(prior['omega'])]:
        st=p.stat();weights.append(dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns))
    manifest=dict(frozen_files=[dict(path=str(p.resolve()),sha256=digest(p)) for p in files],weights=weights)
    if (OUT/'manifest.json').exists() and read(OUT/'manifest.json') != manifest:
        raise ValueError('Existing experiment differs')
    save(OUT/'manifest.json',manifest)
    save(OUT/'status.json',dict(state='prepared',seeds=SEEDS))
    print('PREPARED',OUT,flush=True)


def infer(seeds):
    import torch
    import imageio.v2 as imageio
    from vggtdiff import VGGTDiff

    validate()
    protocol, t = read(OUT/'protocol.json'), read(OUT/'trajectory.json')
    for seed in seeds:
        assert seed in SEEDS
    if any(s != SEEDS[0] for s in seeds):
        if read(OUT/f'seed_{SEEDS[0]}/visual_review.json')['state'] != 'accepted':
            raise ValueError('First seed must be visually reviewed before subsequent seeds')
    pending=[]
    for seed in seeds:
        result=OUT/f'seed_{seed}/result.json'
        if result.exists():verify_rows(read(result)['frames'])
        else:pending.append(seed)
    if not pending:
        print('ALREADY_GENERATED',seeds,flush=True);return
    save(OUT/'status.json',dict(state='loading_diffusion',pending=pending))
    model=VGGTDiff(protocol['checkpoint'],protocol['omega'],base_model_dir=protocol['base'],
        device='cuda',vram_limit_gib=20,training_mode='nvs',load_full_checkpoint_directly=True)
    assert not model.pipe.dit.training
    images=[Image.open(GT/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
    w=np.array(t['source_w2c']+t['target_w2c'],np.float32)
    k=np.array(t['source_intrinsics']+t['target_intrinsics'],np.float32)
    assert w.shape==(14,4,4) and k.shape==(14,3,3)
    for seed in pending:
        root=OUT/f'seed_{seed}';(root/'raw').mkdir(parents=True,exist_ok=True)
        save(root/'status.json',dict(state='generating',seed=seed))
        save(OUT/'status.json',dict(state='generating',seed=seed,pending=pending))
        print('START_SEED',seed,flush=True)
        torch.cuda.reset_peak_memory_stats();start=time.time()
        result=model.generate(images,w,k,height=480,width=832,steps=50,seed=seed)
        assert len(result)==8
        records=[]
        with imageio.get_writer(str(root/'prediction_8.mp4'),fps=2,macro_block_size=16) as writer:
            for i,image in enumerate(result):
                assert image.size==(832,480)
                path=root/'raw'/f'{i:02d}.png';image.save(path);writer.append_data(np.asarray(image))
                records.append(dict(index=i,original_frame=t['target_video_frames'][i],angle=t['target_angles_deg'][i],path=str(path),sha256=digest(path)))
        montage([Path(r['path']) for r in records],[f"{r['index']:02d} | {r['angle']:.2f} deg" for r in records],root/'contact8.jpg',columns=4)
        paths,labels=[],[]
        for i,r in enumerate(records):
            paths += [GT/'gt'/f'{i:02d}.png',Path(r['path'])]
            labels += [f"REAL GT {r['original_frame']}",f"GENERATED seed {seed} | {r['angle']:.2f} deg"]
        montage(paths,labels,root/'gt_vs_generated.jpg',columns=2)
        validate()
        save(root/'result.json',dict(seed=seed,frames=records,elapsed_seconds=time.time()-start,
            max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cameras_sha256=digest(OUT/'trajectory.json'),
            steps=50,dit_eval=not model.pipe.dit.training))
        save(root/'status.json',dict(state='generated_pending_review'))
        print('COMPLETED_SEED',seed,time.time()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','infer'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[SEEDS[0]])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else infer(args.seeds)
