"""64-frame 360 orbit conditioned on the latest real and generated views.

VGGT-Diff takes exactly six references. They are spread around the car:
generated 45, 90 and 150 degrees, plus real 214, 279 and 322 degrees.
Targets reuse the existing near-orbit circle, eight views per batch.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import vggtdiff_controlled_context as c
import vggtdiff_realvideo_unseen as e
from vggtdiff_white_car import digest, save

PARENT = c.PARENT
CONTEXT = c.OUT / 'context8'
REAR = PARENT.with_name('vggtdiff_realvideo_rear_iter1_v1')
ORBIT = PARENT.with_name('vggtdiff_realvideo_orbit360_64_v2_batch8')
OUT = PARENT.with_name('vggtdiff_realvideo_orbit360_64_latest_refs_v1')
SEED = 20261021
STEPS = 50
BATCH = 8


def read(path):
    import json
    return json.loads(Path(path).read_text())


def sources():
    geometry = read(PARENT / 'geometry_check.json')
    parent = read(PARENT / 'trajectory.json')
    context = read(CONTEXT / 'trajectory.json')
    context_result = {float(frame['angle']): frame for frame in read(CONTEXT / 'result.json')['frames']}
    rear = read(REAR / 'trajectory.json')
    rear_result = {float(frame['angle']): frame for frame in read(REAR / 'result.json')['frames']}
    center = np.array(geometry['center'])
    front = np.array(geometry['front'])
    left = np.array(geometry['left'])

    def azimuth(w2c):
        camera = np.linalg.inv(np.array(w2c))[:3, 3]
        horizontal = camera - center
        return float(np.degrees(np.arctan2(-np.dot(horizontal, left), np.dot(horizontal, front))) % 360)

    rows = []
    for index in (5, 2, 0):
        w2c = parent['source_w2c'][index]
        rows.append(dict(role='real', angle=azimuth(w2c), image=PARENT / 'source_views' / f'{index:02d}.png',
                         w2c=w2c, intrinsics=parent['source_intrinsics'][index]))
    for angle, index in ((45.0, 0), (90.0, 4)):
        image = Path(context_result[angle]['path'])
        if digest(image) != context_result[angle]['sha256']:
            raise ValueError(f'Context image changed: {image}')
        if not np.allclose(context['target_w2c'][index], c.trajectory([angle])['target_w2c'][0], atol=1e-8):
            raise ValueError(f'Context {angle:g} camera drifted')
        rows.append(dict(role='generated', angle=angle, image=image, w2c=context['target_w2c'][index],
                         intrinsics=context['target_intrinsics'][index]))
    angle = 150.0
    image = Path(rear_result[angle]['path'])
    if digest(image) != rear_result[angle]['sha256']:
        raise ValueError(f'Rear image changed: {image}')
    rear_index = [float(a) for a in rear['target_angles_deg']].index(angle)
    rows.append(dict(role='generated', angle=azimuth(rear['target_w2c'][rear_index]), image=image,
                     w2c=rear['target_w2c'][rear_index], intrinsics=rear['target_intrinsics'][rear_index]))
    rows.sort(key=lambda row: row['angle'])
    if len(rows) != 6:
        raise ValueError('VGGT-Diff expects six references')
    return rows


def prepare():
    orbit = read(ORBIT / 'trajectory.json')
    angles = np.linspace(0, 360, 64, endpoint=False)
    if not np.allclose(orbit['target_angles_deg'], angles, atol=1e-8):
        raise ValueError('Existing 64-frame orbit is not the 5.625 degree circle')
    chosen = sources()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'source_views').mkdir(exist_ok=True)
    copied = []
    for index, row in enumerate(chosen):
        with Image.open(row['image']) as im:
            rgb = im.convert('RGB')
            if rgb.size != (832, 480):
                raise ValueError(f'{row["image"]} is {rgb.size}')
        destination = OUT / 'source_views' / f'{index:02d}.png'
        if not destination.exists():
            shutil.copyfile(row['image'], destination)
        copied.append(dict(index=index, role=row['role'], angle=row['angle'], original=str(row['image']), sha256=digest(destination)))
    trajectory = dict(source_w2c=[row['w2c'] for row in chosen], source_intrinsics=[row['intrinsics'] for row in chosen],
                      target_w2c=orbit['target_w2c'], target_intrinsics=orbit['target_intrinsics'],
                      target_angles_deg=orbit['target_angles_deg'], intrinsics_image_size=[832, 480],
                      camera_convention='OpenCV W2C')
    save(OUT / 'trajectory.json', trajectory)
    save(OUT / 'sources.json', dict(sources=copied, target_count=64, batch_size=BATCH, seed=SEED, steps=STEPS,
                                    note='Six references spaced around the car from the latest real views and VGGT-Diff views. No 3DGS renders.'))
    print('PREPARED', OUT)
    for item in copied:
        print(f"  {item['role']:9} {item['angle']:7.2f} deg  {Path(item['original']).name}")
    return trajectory


def infer():
    import torch
    from vggtdiff import VGGTDiff

    trajectory = prepare()
    parent = read(PARENT / 'inference_manifest.json')
    checkpoint = Path(parent['checkpoint'])
    if read(checkpoint.with_name('config.json'))['model_family'] != 'nvs':
        raise ValueError('Expected the NVS checkpoint')
    raw = OUT / 'raw'
    batches = OUT / 'batches'
    raw.mkdir(exist_ok=True)
    batches.mkdir(exist_ok=True)
    pending = []
    for batch in range(8):
        result = batches / f'{batch:02d}.json'
        if result.exists():
            for frame in read(result)['frames']:
                if digest(Path(frame['path'])) != frame['sha256']:
                    raise ValueError(f'Batch output changed: {frame["path"]}')
        else:
            pending.append(batch)
    if not pending and (OUT / 'prediction_64.mp4').exists():
        print('ALREADY_COMPLETE', flush=True)
        return
    images = [Image.open(OUT / 'source_views' / f'{i:02d}.png').convert('RGB') for i in range(6)]
    if pending:
        save(OUT / 'status.json', dict(state='loading', pending=pending, seed=SEED, steps=STEPS))
        model = VGGTDiff(parent['checkpoint'], parent['omega'], base_model_dir=parent['base'], device='cuda',
                         vram_limit_gib=20, training_mode='nvs', load_full_checkpoint_directly=True)
        if model.pipe.dit.training:
            raise RuntimeError('Diffusion model is not in eval mode')
        for batch in pending:
            indices = list(range(batch * BATCH, (batch + 1) * BATCH))
            w2c = np.array(trajectory['source_w2c'] + [trajectory['target_w2c'][i] for i in indices], dtype=np.float32)
            intrinsics = np.array(trajectory['source_intrinsics'] + [trajectory['target_intrinsics'][i] for i in indices], dtype=np.float32)
            if w2c.shape != (14, 4, 4):
                raise ValueError(f'Unexpected camera batch {w2c.shape}')
            save(OUT / 'status.json', dict(state='generating', batch=batch + 1, completed_frames=batch * BATCH, target_count=64))
            print(f'START_BATCH {batch + 1}/8', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            outputs = model.generate(images, w2c, intrinsics, height=480, width=832, steps=STEPS, seed=SEED + batch)
            if len(outputs) != BATCH:
                raise ValueError(f'Expected {BATCH} images, got {len(outputs)}')
            records = []
            for index, image in zip(indices, outputs):
                if image.size != (832, 480):
                    raise ValueError(f'Output {index} is {image.size}')
                path = raw / f'{index:02d}.png'
                image.save(path)
                records.append(dict(index=index, angle=trajectory['target_angles_deg'][index], path=str(path), sha256=digest(path)))
            result = dict(batch=batch + 1, seed=SEED + batch, steps=STEPS, frames=records,
                          elapsed_seconds=time.time() - start, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            save(batches / f'{batch:02d}.json', result)
            e.contact([raw / f'{i:02d}.png' for i in indices], [f'{trajectory["target_angles_deg"][i]:.2f}' for i in indices], batches / f'{batch:02d}.jpg', columns=4)
            print(f'COMPLETED_BATCH {batch + 1}/8 {result["elapsed_seconds"]:.1f}s', flush=True)
        del model
        torch.cuda.empty_cache()
    assemble(trajectory)


def assemble(trajectory):
    import imageio.v2 as imageio
    records = []
    overview = Image.new('RGB', (8 * 208, 8 * 148), '#181818')
    draw = ImageDraw.Draw(overview)
    with imageio.get_writer(str(OUT / 'prediction_64.mp4'), fps=8, macro_block_size=16) as writer:
        for index, angle in enumerate(trajectory['target_angles_deg']):
            path = OUT / 'raw' / f'{index:02d}.png'
            with Image.open(path) as im:
                if im.size != (832, 480):
                    raise ValueError(f'{path} is {im.size}')
                panel = Image.new('RGB', (832, 512), '#181818')
                panel.paste(im, (0, 32))
                ImageDraw.Draw(panel).text((12, 10), f'latest refs | {index:02d}/63 | {angle:.2f} deg', fill='white')
                writer.append_data(np.array(panel))
                x, y = index % 8 * 208, index // 8 * 148
                overview.paste(im.resize((208, 120), Image.Resampling.LANCZOS), (x, y + 28))
                draw.text((x + 5, y + 8), f'{index:02d} | {angle:.1f}', fill='white')
            records.append(dict(index=index, angle=angle, path=str(path), sha256=digest(path)))
    overview.save(OUT / 'overview_64.jpg', quality=95)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(OUT / 'prediction_64.mp4')]))
    if int(probe['streams'][0]['nb_read_frames']) != 64:
        raise ValueError('Orbit video does not have 64 frames')
    save(OUT / 'result.json', dict(frames=records, video=str(OUT / 'prediction_64.mp4')))
    save(OUT / 'status.json', dict(state='completed', completed_frames=64, target_count=64))
    print('COMPLETED_ALL_64', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        return
    infer()


if __name__ == '__main__':
    main()
