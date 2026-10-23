"""One VGGT-Diff batch for the rear-right gap.

Six references: four real video views plus the generated 90° and 101.25° images.
Five targets, denoised together: 112.5°, 120°, 127.5°, 135°, 150°.
Does not train 3DGS and does not use rendered Gaussians as references.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import vggtdiff_controlled_context as c
import vggtdiff_realvideo_unseen as e
from vggtdiff_white_car import digest, save

PARENT = c.PARENT
CONTEXT = c.OUT / 'context8'
OUT = PARENT.with_name('vggtdiff_realvideo_rear_iter1_v1')
REAL_INDICES = (0, 2, 4, 5)
GENERATED_ANGLES = (90.0, 101.25)
TARGET_ANGLES = (112.5, 120.0, 127.5, 135.0, 150.0)
SEED = 20261021
STEPS = 50


def azimuth(position, center, front, left):
    horizontal = np.asarray(position) - np.asarray(center)
    return math.degrees(math.atan2(-float(np.dot(horizontal, left)), float(np.dot(horizontal, front)))) % 360


def load_inputs():
    geometry = c.read(PARENT / 'geometry_check.json')
    parent_trajectory = c.read(PARENT / 'trajectory.json')
    context_trajectory = c.read(CONTEXT / 'trajectory.json')
    context_result = c.read(CONTEXT / 'result.json')
    center = np.array(geometry['center'])
    front = np.array(geometry['front'])
    left = np.array(geometry['left'])
    up = np.array(geometry['up'])
    sources = []
    for index in REAL_INDICES:
        image = PARENT / 'source_views' / f'{index:02d}.png'
        w2c = np.array(parent_trajectory['source_w2c'][index])
        camera = np.linalg.inv(w2c)[:3, 3]
        angle = azimuth(camera, center, front, left)
        expected = azimuth(geometry['source_centers'][index], center, front, left)
        if abs(angle - expected) > 0.05:
            raise ValueError(f'Real camera {index} azimuth {angle:.2f} != {expected:.2f}')
        sources.append(dict(role='real', angle=angle, image=image, w2c=w2c,
                            intrinsics=np.array(parent_trajectory['source_intrinsics'][index])))
    by_angle = {float(frame['angle']): frame for frame in context_result['frames']}
    context_angles = [float(a) for a in context_trajectory['target_angles_deg']]
    for angle in GENERATED_ANGLES:
        frame = by_angle[angle]
        image = Path(frame['path'])
        if digest(image) != frame['sha256']:
            raise ValueError(f'Generated reference changed: {image}')
        index = context_angles.index(angle)
        stored = np.array(context_trajectory['target_w2c'][index])
        rebuilt = c.trajectory([angle])['target_w2c'][0]
        if not np.allclose(stored, rebuilt, rtol=0, atol=1e-12):
            raise ValueError(f'Generated {angle:g} camera does not match the saved trajectory')
        sources.append(dict(role='generated', angle=angle, image=image, w2c=stored,
                            intrinsics=np.array(context_trajectory['target_intrinsics'][index])))
    if len(sources) != 6:
        raise ValueError('VGGT-Diff expects exactly six references')
    images = []
    for item in sources:
        with Image.open(item['image']) as im:
            rgb = im.convert('RGB')
            if rgb.size != (832, 480):
                raise ValueError(f'{item["image"]} is {rgb.size}, expected 832x480')
            images.append(rgb.copy())
    targets = c.trajectory(list(TARGET_ANGLES))
    for angle, w2c in zip(TARGET_ANGLES, targets['target_w2c']):
        if angle in context_angles:
            index = context_angles.index(angle)
            if not np.allclose(w2c, context_trajectory['target_w2c'][index], rtol=0, atol=1e-12):
                raise ValueError(f'Target {angle:g} left the existing right-side orbit')
        camera = np.linalg.inv(np.array(w2c))[:3, 3]
        got = azimuth(camera, center, front, left)
        delta = min(abs(got - angle), 360 - abs(got - angle))
        if delta > 0.05:
            raise ValueError(f'Target azimuth {got:.3f} != requested {angle:g}')
    lo, hi = np.array(geometry['observed_car_bounds_local'])
    origin = np.array(geometry['ground_origin'])
    axes = np.stack([front, left, up], axis=1)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = corners @ axes.T + origin
    k_target = np.array(targets['target_intrinsics'][0])
    bounds = []
    for angle, w2c in zip(TARGET_ANGLES, targets['target_w2c']):
        uv, depth = e.project(corners, np.array(w2c), k_target)
        bounds.append(dict(angle=angle, min_depth=float(depth.min()),
                           pixels=[float(x) for x in (*uv.min(0), *uv.max(0))]))
        if depth.min() <= 0:
            raise ValueError(f'Target {angle:g} sees the observed car behind the camera')
    known = {float(a): box for a, box in zip(geometry['target_angles_deg'], geometry['target_bounds_model_pixels'])}
    for item in bounds:
        if item['angle'] in known and not np.allclose(item['pixels'], known[item['angle']], atol=0.2):
            raise ValueError(f"Target {item['angle']:g} framing drifted from the original orbit")
    return sources, images, targets, bounds


def prepare():
    sources, images, targets, bounds = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'source_views').mkdir(exist_ok=True)
    copied = []
    for i, (item, image) in enumerate(zip(sources, images)):
        path = OUT / 'source_views' / f'{i:02d}.png'
        image.save(path)
        copied.append(dict(index=i, role=item['role'], angle_degrees=item['angle'],
                           original=str(item['image']), sha256=digest(path)))
    trajectory = dict(
        source_w2c=[item['w2c'].tolist() for item in sources],
        source_intrinsics=[item['intrinsics'].tolist() for item in sources],
        target_w2c=targets['target_w2c'],
        target_intrinsics=targets['target_intrinsics'],
        target_angles_deg=list(TARGET_ANGLES),
        intrinsics_image_size=[832, 480],
        camera_convention='OpenCV W2C',
        observed_car_bounds_pixels=bounds,
    )
    save(OUT / 'trajectory.json', trajectory)
    save(OUT / 'sources.json', dict(
        references='4 real video views + generated 90 and 101.25 only; no 112.5/135 and no 3DGS render',
        sources=copied, targets=list(TARGET_ANGLES), seed=SEED, steps=STEPS))
    return sources, images, trajectory


def infer():
    import torch
    from vggtdiff import VGGTDiff

    if (OUT / 'result.json').exists():
        print('ALREADY_COMPLETE', OUT / 'result.json', flush=True)
        return
    sources, images, trajectory = prepare()
    parent = c.read(PARENT / 'inference_manifest.json')
    checkpoint = Path(parent['checkpoint'])
    config = c.read(checkpoint.with_name('config.json'))
    if config['model_family'] != 'nvs':
        raise ValueError(f"Expected the NVS checkpoint, got {config['model_family']}")
    w2c = np.array(trajectory['source_w2c'] + trajectory['target_w2c'], dtype=np.float32)
    intrinsics = np.array(trajectory['source_intrinsics'] + trajectory['target_intrinsics'], dtype=np.float32)
    if w2c.shape != (11, 4, 4) or intrinsics.shape != (11, 3, 3):
        raise ValueError(f'Unexpected camera batch {w2c.shape} {intrinsics.shape}')
    save(OUT / 'status.json', dict(state='loading', targets=list(TARGET_ANGLES), seed=SEED, steps=STEPS))
    model = VGGTDiff(parent['checkpoint'], parent['omega'], base_model_dir=parent['base'],
                     device='cuda', vram_limit_gib=20, training_mode='nvs',
                     load_full_checkpoint_directly=True)
    if model.pipe.dit.training:
        raise RuntimeError('Diffusion model is not in eval mode')
    save(OUT / 'status.json', dict(state='generating', targets=list(TARGET_ANGLES), seed=SEED, steps=STEPS))
    print('START_REAR_ITER1', list(TARGET_ANGLES), 'gpu', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = model.generate(images, w2c, intrinsics, height=480, width=832, steps=STEPS, seed=SEED)
    if len(outputs) != len(TARGET_ANGLES):
        raise ValueError(f'Expected {len(TARGET_ANGLES)} images, got {len(outputs)}')
    raw = OUT / 'raw'
    raw.mkdir(exist_ok=True)
    frames = []
    for i, (angle, image) in enumerate(zip(TARGET_ANGLES, outputs)):
        if image.size != (832, 480):
            raise ValueError(f'Output {i} is {image.size}')
        path = raw / f'{i:02d}.png'
        image.save(path)
        frames.append(dict(index=i, angle=angle, path=str(path), sha256=digest(path)))
        print('WROTE', path, angle, flush=True)
    e.contact([raw / f'{i:02d}.png' for i in range(len(frames))],
              [f'{angle:g} deg' for angle in TARGET_ANGLES], OUT / 'predictions_contact.jpg', columns=5)
    result = dict(frames=frames, elapsed_seconds=time.time() - start,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  seed=SEED, steps=STEPS, references=[item['role'] for item in sources],
                  note='Generation only. Mask width still has to be checked before any 3DGS training.')
    save(OUT / 'result.json', result)
    save(OUT / 'status.json', dict(state='completed_pending_width_check', frames=len(frames)))
    print('COMPLETED_REAR_ITER1', json_elapsed(result), flush=True)


def json_elapsed(result):
    return f"{result['elapsed_seconds']:.1f}s peak {result['peak_allocated_gib']:.2f} GiB"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        sources, _, trajectory = prepare()
        print('PREPARED', OUT)
        for item in sources:
            print(f"  {item['role']:9} {item['angle']:7.2f} deg  {item['image'].name}")
        for item in trajectory['observed_car_bounds_pixels']:
            print(f"  target {item['angle']:7.2f}  bounds {['%.1f' % v for v in item['pixels']]}  depth {item['min_depth']:.3f}")
        return
    infer()


if __name__ == '__main__':
    main()
