"""Compare completed batch8 variants with frozen targets and vehicle-only metrics.

The source cameras/composition differ; this is not an isolated reference-pixel
ablation. Run prepare, inspect mask_review.jpg, record mask_review.json, then evaluate.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import itertools
import json

import cv2
import numpy as np
from PIL import Image

import crossview_geometry_eval as geom
import crossview_geometry_dense_gt as dense
import crossview_geometry_matched_eval as matched
import vggtdiff_batch08_real_anchors as job

OUT = job.OUT / 'analysis_vs_generated_v1'
RUNS = {'all_generated': job.BASELINE, 'real4_gen2': job.OUT}


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    job.chain.save(p, value)


def choose_vehicle(segmenter, rgb):
    import torch
    with torch.inference_mode():
        result = segmenter.segment_candidates(
            torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda() / 255,
            'white car')
    choices = []
    for c in ([] if result is None else result.candidates):
        mask = c.mask.cpu().numpy() > .5
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        area = float(mask.mean())
        below = float(mask[140:].sum() / mask.sum())
        center = np.array([xs.mean(), ys.mean()])
        score = float(c.score) + 2 * area + below - .0008 * np.linalg.norm(center - [416, 330])
        # Identical selector in both variants; frontal car occupies less area.
        if .02 < area < .7 and below > .8:
            choices.append((score, mask))
    if not choices:
        raise RuntimeError('No eligible vehicle mask')
    return max(choices, key=lambda c: c[0])[1]


def prepare():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    OUT.mkdir(exist_ok=True)
    trajectories = {}
    for key, root in RUNS.items():
        _, trajectories[key] = job.chain.validate(root)
    for key in ['target_w2c', 'target_intrinsics', 'target_angles_deg']:
        np.testing.assert_array_equal(trajectories['all_generated'][key], trajectories['real4_gen2'][key])
    manifests = {key: read(root / 'manifest.json') for key, root in RUNS.items()}
    for key in ['seed', 'steps', 'checkpoint', 'weights']:
        assert manifests['all_generated'][key] == manifests['real4_gen2'][key], key
    protocol = dict(
        comparison='Two existing single-seed batch8 outputs; no new diffusion or reconstruction',
        image_size=[832, 480], same_target_cameras=True, same_seed_steps_weights=True,
        source_camera_composition_is_different=True,
        region='SAM3 white-car masks; identical selector, area .02-.7, >80% below y140; visual review required',
        matcher='SIFT12000, mutual Lowe ratio .75; no epipolar or RANSAC prefilter',
        support_minimum_matches=30, primary_pair_gap_degrees=15,
        primary_pair_group='13 within-batch8 pairs <=15deg per run',
        boundary_group='Three batch7->batch8 pairs <=15deg; batch7 images54/55 reused conditioning references',
        triples='Six within-batch8 consecutive triples and two boundary triples; raw and direct descriptor cycle; min parallax 1deg, sensitivities .5/2',
        interpretation='Fixed requested-camera compatibility, not actual pose GT or proof of 3D accuracy; no independent same-camera real GT for batch8; support collapse is reported',
        acceptance_threshold=None)
    save(OUT / 'protocol.json', protocol)
    b7t = read(job.B7 / 'trajectory.json')
    b7r = read(job.B7 / 'result.json')
    rows = []
    overlays, labels = [], []
    segmenter = None
    for variant, root in RUNS.items():
        t = trajectories[variant]
        result = read(root / 'result.json')
        assert result['cameras_sha256'] == job.chain.digest(root / 'trajectory.json')
        for i in range(10):
            ident = 54 + i
            local = i + 6 if i < 2 else i - 2
            r = b7r['frames'][local] if i < 2 else result['frames'][local]
            cam = b7t if i < 2 else t
            image = Path(r['path'])
            assert job.chain.digest(image) == r['sha256']
            rgb = np.asarray(Image.open(image).convert('RGB'))
            assert rgb.shape == (480, 832, 3)
            mask_path = OUT / variant / 'masks' / f'{ident}.png'
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not mask_path.exists():
                if i < 2:
                    job.chain.frozen_copy(job.OUT / 'input_geometry_check' / f'{i:02d}_mask.png', mask_path)
                else:
                    if segmenter is None:
                        segmenter = SAM3Segmenter(SimpleNamespace(
                            source_dir=str(job.chain.REPO / 'thirdparty/sam3'),
                            checkpoint_path=str(job.chain.REPO / 'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
                            device='cuda', confidence_threshold=.15, instance_strategy='best_score'))
                    mask = choose_vehicle(segmenter, rgb)
                    Image.fromarray(mask.astype('uint8') * 255).save(mask_path)
            mask = geom.read_mask(mask_path) > 0
            ys, xs = np.where(mask)
            assert len(xs) > 1000
            row = dict(index=ident, original_frame=ident, variant=variant, image=str(image), mask=str(mask_path),
                       sha256=r['sha256'], mask_sha256=job.chain.digest(mask_path),
                       K=cam['target_intrinsics'][local], T=cam['target_w2c'][local],
                       azimuth_deg=cam['target_angles_deg'][local],
                       mask_area=int(mask.sum()), mask_bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)])
            rows.append(row)
            overlay = rgb.copy()
            overlay[mask] = (.55 * rgb[mask] + .45 * np.array([0, 255, 0])).astype('uint8')
            dest = mask_path.with_name(f'{ident}_overlay.jpg')
            Image.fromarray(overlay).save(dest)
            if i >= 2:
                overlays.append(dest)
                labels.append(f'{variant} | {ident} | {row["azimuth_deg"]:.3f} deg')
            print('MASK', variant, ident, row['mask_area'], flush=True)
        cap = cv2.VideoCapture(str(root / 'prediction_8.mp4'))
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        assert decoded == 8
    del segmenter
    torch.cuda.empty_cache()
    job.chain.montage(overlays, labels, OUT / 'mask_review.jpg', columns=4)
    save(OUT / 'frames.json', dict(frames=rows))
    files = [Path(__file__).resolve(), Path(geom.__file__).resolve(), Path(dense.__file__).resolve(), Path(matched.__file__).resolve(), OUT / 'frames.json', OUT / 'protocol.json']
    files += [p / name for p in RUNS.values() for name in ['manifest.json', 'trajectory.json', 'result.json']]
    files += list(dict.fromkeys(Path(r[k]) for r in rows for k in ['image', 'mask']))
    save(OUT / 'manifest.json', dict(files=[dict(path=str(p), sha256=job.chain.digest(p)) for p in files]))
    for p in [Path(__file__).resolve(), Path(geom.__file__).resolve(), Path(dense.__file__).resolve(), Path(matched.__file__).resolve()]:
        job.chain.frozen_copy(p, OUT / 'implementation' / p.name)
    save(OUT / 'status.json', dict(state='prepared_pending_mask_review'))


def validate():
    for r in read(OUT / 'manifest.json')['files']:
        assert job.chain.digest(Path(r['path'])) == r['sha256'], r['path']


def evaluate():
    validate()
    assert read(OUT / 'mask_review.json')['state'] == 'accepted'
    cv2.setNumThreads(4)
    all_meta = read(OUT / 'frames.json')['frames']
    summaries = {}
    for variant in RUNS:
        root = OUT / variant
        frames = [matched.feature_frame(r, root / 'features') for r in all_meta if r['variant'] == variant]
        matches = matched.build_matches(frames, root)
        by_id = {f['original_frame']: f for f in frames}
        rows = []
        for (a, b), (ia, ib) in matches.items():
            fa, fb = by_id[a], by_id[b]
            r = geom.pair_record(a, b, fa['xy'][ia], fb['xy'][ib], fa['mask'], fb['mask'], fa['K'], fa['T'], fb['K'], fb['T'])
            r.update(azimuth_gap_deg=geom.angular_gap(fa['azimuth_deg'], fb['azimuth_deg']),
                     group='within_batch8' if a >= 56 else ('boundary' if b >= 56 else 'batch7_reference'), index_gap=b-a)
            rows.append(r)
        geom.write_csv(root / 'pairs.csv', rows, geom.PAIR_FIELDS + ['group', 'index_gap'])
        geom._write_overlays(root, frames, matches, rows)
        edges = {k: set(zip(v[0].tolist(), v[1].tolist())) for k, v in matches.items()}
        working = {i: dict(K=f['K'], T=f['T'], xy=f['xy'], mask=f['mask']) for i, f in by_id.items()}
        triples = []
        attempts_by_group = {g: dict(raw=[], cycle=[]) for g in ['within_batch8', 'boundary']}
        for start in range(54, 62):
            ids = list(range(start, start+3))
            group = 'within_batch8' if start >= 56 else 'boundary'
            sub = {k: v for k, v in matches.items() if set(k).issubset(ids)}
            tracks = [t for t in geom.build_tracks(sub) if t['n_views'] >= 3]
            raw, cycle = [], []
            for track in tracks:
                track = dict(track, track_id=start*100000+track['track_id'])
                attempts = [a for threshold in [.5, 1, 2] for a in geom.heldout_attempts(track, working, threshold)]
                raw += attempts
                if dense.direct_cycle(track, edges):
                    cycle += attempts
            attempts_by_group[group]['raw'] += raw
            attempts_by_group[group]['cycle'] += cycle
            triples.append(dict(frames=ids, group=group, raw=dense.summarized_attempts(raw), cycle=dense.summarized_attempts(cycle)))
        pairs = {}
        for group in ['within_batch8', 'boundary']:
            sub = [r for r in rows if r['group'] == group]
            pairs[group] = dict(all=dense.pair_summary(sub), primary15=dense.pair_summary([r for r in sub if r['azimuth_gap_deg'] <= 15]), adjacent=dense.pair_summary([r for r in sub if r['index_gap'] == 1]))
        summaries[variant] = dict(pair_groups=pairs, triples=triples,
            track_groups={g: {k: dense.summarized_attempts(v) for k, v in d.items()} for g, d in attempts_by_group.items()},
            adjacent_pairs=[{k: v for k, v in r.items() if k != 'errors'} for r in rows if r['index_gap'] == 1])
        save(root / 'summary.json', summaries[variant])
        for group, attempts in attempts_by_group.items():
            for method, values in attempts.items():
                geom.write_csv(root / f'tracks_{group}_{method}.csv', values, matched.TRACK_FIELDS)
        print('SUMMARY', variant, json.dumps(summaries[variant]['pair_groups']), flush=True)
    save(OUT / 'summary.json', summaries)
    validate()
    save(OUT / 'audit.json', dict(state='passed', input_hashes_verified=True, identical_target_cameras=True,
        same_seed_steps_weights=True, videos_decoded_frames=8, masks_visually_reviewed=True,
        no_camera_refit=True, no_epipolar_or_ransac_prefilter=True,
        quality_acceptance_threshold=None, same_camera_real_GT=False))
    save(OUT / 'status.json', dict(state='evaluated_pending_interpretation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        evaluate()
