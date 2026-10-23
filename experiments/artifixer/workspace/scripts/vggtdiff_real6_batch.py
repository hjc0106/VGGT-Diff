"""Generate one existing orbit batch using only the six real references."""
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

REAL = BASE / 'vggtdiff_realvideo_unseen_right_near_v2'
ORBIT = BASE / 'vggtdiff_realvideo_orbit360_64_latest_refs_v1'


def read(path):
    return json.loads(path.read_text())


def prepare(args):
    real, orbit = read(REAL / 'trajectory.json'), read(ORBIT / 'trajectory.json')
    geometry = read(REAL / 'geometry_check.json')
    parent = read(REAL / 'inference_manifest.json')
    indices = list(range((args.batch - 1) * 8, args.batch * 8))
    angles = [orbit['target_angles_deg'][i] for i in indices]
    if not np.allclose(angles, np.array(indices) * 5.625):
        raise ValueError('Orbit angle/index mismatch')
    output = args.output or BASE / f'vggtdiff_real6_orbit_batch{args.batch:02d}_v1'
    if output.resolve() in {REAL.resolve(), ORBIT.resolve()}:
        raise ValueError('Output must be separate from source experiments')
    seed = args.seed if args.seed is not None else 20261021 + args.batch - 1
    trajectory = dict(source_w2c=real['source_w2c'], source_intrinsics=real['source_intrinsics'],
                      target_w2c=[orbit['target_w2c'][i] for i in indices],
                      target_intrinsics=[orbit['target_intrinsics'][i] for i in indices],
                      target_angles_deg=angles, orbit_indices=indices,
                      intrinsics_image_size=[832, 480], camera_convention='OpenCV W2C')
    files = [REAL / 'trajectory.json', ORBIT / 'trajectory.json',
             REAL / 'inference_manifest.json', REAL / 'geometry_check.json', Path(__file__).resolve()]
    sources = []
    for i, w in enumerate(real['source_w2c']):
        source = REAL / 'source_views' / f'{i:02d}.png'
        with Image.open(source) as image:
            if image.size != (832, 480):
                raise ValueError(f'Unexpected source dimensions: {source}')
        delta = np.linalg.inv(np.array(w))[:3, 3] - np.array(geometry['center'])
        angle = float(np.degrees(np.arctan2(-delta @ np.array(geometry['left']),
                                          delta @ np.array(geometry['front']))) % 360)
        sources.append(dict(index=i, role='real', original=str(source), angle=angle,
                            sha256=digest(source)))
        files.append(source)
    config = read(Path(parent['checkpoint']).with_name('config.json'))
    if config['model_family'] != 'nvs':
        raise ValueError('Expected an NVS checkpoint')
    manifest = dict(batch=args.batch, indices=indices, angles=angles, sources=sources,
                    seed=seed, steps=args.steps, resolution=[832, 480], training_mode='nvs',
                    vram_limit_gib=args.vram_limit_gib,
                    checkpoint=parent['checkpoint'], omega=parent['omega'], base=parent['base'],
                    frozen_files=[dict(path=str(p), sha256=digest(p)) for p in files])
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('manifest.json', manifest), ('trajectory.json', trajectory)]:
        path = output / name
        if path.exists() and read(path) != value:
            raise ValueError(f'Existing experiment differs: {path}; choose a new --output')
        save(path, value)
    for source in sources:
        dst = output / 'source_views' / f"{source['index']:02d}.png"
        dst.parent.mkdir(exist_ok=True)
        if not dst.exists():
            shutil.copyfile(source['original'], dst)
        if digest(dst) != source['sha256']:
            raise ValueError(f'Copied input changed: {dst}')
    print(json.dumps(dict(output=str(output), batch=args.batch, indices=indices, angles=angles,
                          real_reference_angles=[r['angle'] for r in sources], seed=seed), indent=2), flush=True)
    return output, trajectory, manifest


def infer(output, trajectory, manifest):
    import torch
    import imageio.v2 as imageio
    from vggtdiff import VGGTDiff
    from vggtdiff_latest_coverage_ab import montage

    result_path = output / 'result.json'
    if result_path.exists():
        raise FileExistsError(f'Completed output exists: {result_path}')
    for row in manifest['frozen_files']:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f"Input changed: {row['path']}")
    save(output / 'status.json', dict(state='loading', batch=manifest['batch']))
    model = VGGTDiff(manifest['checkpoint'], manifest['omega'], base_model_dir=manifest['base'],
                     device='cuda', vram_limit_gib=manifest['vram_limit_gib'],
                     training_mode='nvs', load_full_checkpoint_directly=True)
    images = [Image.open(output / 'source_views' / f'{i:02d}.png').convert('RGB') for i in range(6)]
    w = np.asarray(trajectory['source_w2c'] + trajectory['target_w2c'], dtype=np.float32)
    k = np.asarray(trajectory['source_intrinsics'] + trajectory['target_intrinsics'], dtype=np.float32)
    if w.shape != (14, 4, 4) or model.pipe.dit.training:
        raise ValueError('Unexpected camera count or training state')
    save(output / 'status.json', dict(state='generating', batch=manifest['batch']))
    start = time.time()
    predictions = model.generate(images, w, k, height=480, width=832,
                                 steps=manifest['steps'], seed=manifest['seed'])
    if len(predictions) != 8:
        raise ValueError(f'Expected eight predictions, got {len(predictions)}')
    raw = output / 'raw'
    raw.mkdir(exist_ok=True)
    records, paths, labels = [], [], []
    with imageio.get_writer(str(output / 'prediction_8.mp4'), fps=2, macro_block_size=16) as writer:
        for image, index, angle in zip(predictions, manifest['indices'], manifest['angles']):
            if image.size != (832, 480):
                raise ValueError(f'Unexpected output dimensions: {image.size}')
            path = raw / f'{index:02d}.png'
            image.save(path)
            writer.append_data(np.asarray(image))
            paths.append(path)
            labels.append(f'real6 | {index:02d} | {angle:g} deg')
            records.append(dict(index=index, angle=angle, path=str(path), sha256=digest(path)))
    montage(paths, labels, output / 'contact8.jpg', columns=4)
    pair_paths, pair_labels = [], []
    for r in records:
        pair_paths.extend([ORBIT / 'raw' / f"{r['index']:02d}.png", Path(r['path'])])
        pair_labels.extend([f"mixed refs | {r['angle']:g} deg", f"real6 refs | {r['angle']:g} deg"])
    montage(pair_paths, pair_labels, output / 'mixed_vs_real6.jpg', columns=2)
    save(result_path, dict(frames=records, elapsed_seconds=time.time() - start,
                           video=str(output / 'prediction_8.mp4')))
    save(output / 'status.json', dict(state='completed_pending_visual_review', completed_frames=8))
    print('COMPLETED', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=range(1, 9), default=6)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--vram-limit-gib', type=float, default=20)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output, trajectory, manifest = prepare(args)
    if not args.prepare_only:
        infer(output, trajectory, manifest)


if __name__ == '__main__':
    main()
