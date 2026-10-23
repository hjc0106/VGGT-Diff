"""Rebuild the fifteen-image set from one jointly generated eight-view batch.

The previous mixed15 supervision stitched group A (45-101.25) to group B
(112.5-135). This experiment keeps the six real views and replaces every
generated view with context8, which was denoised together. 123.75 exists only
in group B, so it is left out. Cameras, initialization and the warmup
checkpoint stay the ones used by mixed15.
"""
import shutil, struct, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from vggtdiff_white_car import BASE, REPO, digest, save
import vggtdiff_realvideo_unseen as e

OLD = BASE / 'vggtdiff_car15_reconstruction_v1'
GROUP = BASE / 'vggtdiff_realvideo_controlled_context_v1/context8'
OUT = BASE / 'vggtdiff_car15_groupa_v1'


def read(path):
    import json
    return json.loads(path.read_text())


def same_camera(a, b):
    return np.allclose(np.array(a), np.array(b), atol=1e-6, rtol=0)


def collect_rows():
    old = {round(float(row['angle']), 2): row for row in read(OLD / 'inputs.json')['frames'] if row['kind'] == 'generated'}
    real = [row for row in read(OLD / 'inputs.json')['frames'] if row['kind'] == 'real']
    generated = read(GROUP / 'result.json')['frames']
    trajectory = read(GROUP / 'trajectory.json')
    assert [row['angle'] for row in generated] == trajectory['target_angles_deg']
    rows = []
    for source in real:
        rows.append(dict(source))
    for index, frame in enumerate(generated):
        angle = round(float(frame['angle']), 2)
        previous = old[angle]
        w2c = trajectory['target_w2c'][index]
        intrinsics = trajectory['target_intrinsics'][index]
        assert same_camera(w2c, previous['w2c']) and same_camera(intrinsics, previous['k'])
        image = Path(frame['path'])
        assert digest(image) == frame['sha256']
        reused = digest(image) == previous['sha256']
        if angle <= 101.25:
            assert reused, angle
        else:
            assert not reused, angle
        rows.append(dict(index=6 + index, name=f'{6 + index:02d}.png', kind='generated', source=str(image),
                          sha256=frame['sha256'], w2c=w2c, k=intrinsics, angle=angle,
                          previous_name=previous['name'], reused_previous_image=reused))
    assert [row['angle'] for row in rows if row['kind'] == 'generated'] == [45.0, 56.25, 67.5, 78.75, 90.0, 101.25, 112.5, 135.0]
    return rows


def choose_mask(segmenter, image):
    import torch
    with torch.inference_mode():
        candidates = segmenter.segment_candidates(torch.from_numpy(image.copy()).permute(2, 0, 1).float().cuda() / 255, 'white car')
    choices = []
    for candidate in ([] if candidates is None else candidates.candidates):
        mask = candidate.mask.cpu().numpy() > .5
        area = mask.mean()
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        below = mask[140:].sum() / max(1, mask.sum())
        center = np.array([xs.mean(), ys.mean()])
        score = float(candidate.score) + 2 * area + below - .0008 * np.linalg.norm(center - [416, 330])
        if .05 < area < .7 and below > .8:
            choices.append((score, mask))
    if not choices:
        raise RuntimeError('No eligible white-car mask')
    return max(choices, key=lambda item: item[0])[1]


def write_colmap(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    images = root / 'images'
    images.mkdir(exist_ok=True)
    sparse = root / 'sparse/0'
    sparse.mkdir(parents=True, exist_ok=True)
    cameras = []
    records = []
    for camera_id, row in enumerate(rows, start=1):
        intrinsics = np.array(row['k'])
        pose = np.array(row['w2c'])
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
        quaternion = np.r_[quaternion[3], quaternion[:3]]
        cameras.append(f'{camera_id} OPENCV 832 480 {intrinsics[0, 0]:.17g} {intrinsics[1, 1]:.17g} {intrinsics[0, 2]:.17g} {intrinsics[1, 2]:.17g} 0 0 0 0')
        records.append(' '.join(map(str, [camera_id, *quaternion, *pose[:3, 3], camera_id, row['name']])))
        shutil.copyfile(OUT / 'images' / row['name'], images / row['name'])
        shutil.copyfile(OUT / 'masks' / row['name'], images / f'{Path(row["name"]).stem}_mask.png')
    (sparse / 'cameras.txt').write_text('\n'.join(cameras) + '\n')
    (sparse / 'images.txt').write_text('\n\n'.join(records) + '\n\n')
    (sparse / 'points3D.txt').write_text('# Shared with the previous experiment\n')
    with (sparse / 'cameras.bin').open('wb') as handle:
        handle.write(struct.pack('<Q', len(rows)))
        for camera_id, row in enumerate(rows, start=1):
            intrinsics = np.array(row['k'])
            handle.write(struct.pack('<iiQQ', camera_id, 4, 832, 480))
            handle.write(struct.pack('<8d', intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2], 0, 0, 0, 0))
    with (sparse / 'images.bin').open('wb') as handle:
        handle.write(struct.pack('<Q', len(rows)))
        for camera_id, row in enumerate(rows, start=1):
            pose = np.array(row['w2c'])
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            quaternion = np.r_[quaternion[3], quaternion[:3]]
            handle.write(struct.pack('<i7di', camera_id, *quaternion, *pose[:3, 3], camera_id))
            handle.write(row['name'].encode() + b'\0')
            handle.write(struct.pack('<Q', 0))
    shutil.copyfile(OLD / 'initial_points.ply', root / 'point_cloud.ply')


def main():
    import torch
    sys.path.insert(0, str(REPO / 'worldact'))
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    rows = collect_rows()
    for directory in [OUT / 'images', OUT / 'masks', OUT / 'rgba']:
        directory.mkdir(parents=True, exist_ok=True)
    segmenter = None
    stats = []
    tiles = []
    for row in rows:
        source = Path(row['source']) if row['kind'] == 'generated' else OLD / 'images' / row['name']
        if row['kind'] == 'real':
            assert digest(source) == row['sha256']
        destination = OUT / 'images' / row['name']
        if not destination.exists():
            shutil.copyfile(source, destination)
        assert digest(destination) == digest(source)
        mask_path = OUT / 'masks' / row['name']
        if row['kind'] == 'real' or row['reused_previous_image']:
            previous = row['name'] if row['kind'] == 'real' else row['previous_name']
            if not mask_path.exists():
                shutil.copyfile(OLD / 'masks' / previous, mask_path)
        elif not mask_path.exists():
            if segmenter is None:
                segmenter = SAM3Segmenter(SimpleNamespace(source_dir=str(REPO / 'thirdparty/sam3'), checkpoint_path=str(REPO / 'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'), device='cuda', confidence_threshold=.15, instance_strategy='best_score'))
            image = np.array(Image.open(destination).convert('RGB'))
            mask = choose_mask(segmenter, image)
            Image.fromarray(mask.astype('uint8') * 255).save(mask_path)
        image = np.array(Image.open(destination).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')) > 127
        ys, xs = np.where(mask)
        assert len(xs) > 1000 and mask.shape == (480, 832)
        rgba = np.dstack([image, mask.astype('uint8') * 255])
        Image.fromarray(rgba).save(OUT / 'rgba' / row['name'])
        overlay = image.copy()
        overlay[mask] = (image[mask] * .6 + np.array([0, 255, 0]) * .4).astype('uint8')
        tile = Image.new('RGB', (416, 272), '#181818')
        tile.paste(Image.fromarray(overlay).resize((416, 240)), (0, 32))
        label = f"{row['name']} {row['kind']} {row.get('angle', 'real')} | {mask.mean():.3f}"
        ImageDraw.Draw(tile).text((8, 9), label, fill='white')
        tiles.append(tile)
        stats.append(dict(name=row['name'], kind=row['kind'], angle=row.get('angle'), area=float(mask.mean()), width=int(xs.max() - xs.min()), center=[float(xs.mean()), float(ys.mean())], sha256=digest(mask_path)))
        print('MASK', stats[-1], flush=True)
    sheet = Image.new('RGB', (416 * 3, 272 * 5))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index % 3 * 416, index // 3 * 272))
    sheet.save(OUT / 'masks_contact.jpg', quality=95)
    serializable = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != 'reused_previous_image'}
        serializable.append(item)
    save(OUT / 'inputs.json', dict(frames=serializable, width=832, height=480, coordinates='Original VGGT-Omega world; OpenCV W2C', generated_source=str(GROUP), dropped_angle=123.75, reason='123.75 was produced only by the conflicting group-B batch'))
    save(OUT / 'masks.json', dict(frames=stats))
    shutil.copyfile(OLD / 'initial_points.ply', OUT / 'initial_points.ply')
    shutil.copyfile(OLD / 'initialization.json', OUT / 'initialization.json')
    shutil.copyfile(OLD / 'warmup_checkpoint.pt', OUT / 'warmup_checkpoint.pt')
    assert digest(OUT / 'initial_points.ply') == digest(OLD / 'initial_points.ply')
    assert digest(OUT / 'warmup_checkpoint.pt') == digest(OLD / 'warmup_checkpoint.pt')
    write_colmap(OUT / 'datasets/groupa', rows)
    e.contact([OUT / 'rgba' / row['name'] for row in rows if row['kind'] == 'generated'], [f"{row['angle']:.2f} deg" for row in rows if row['kind'] == 'generated'], OUT / 'groupa_views.jpg', 4)
    save(OUT / 'preparation.json', dict(state='prepared', real_images=6, generated_images=8, shared_initialization=str(OLD / 'initial_points.ply'), shared_warmup=str(OLD / 'warmup_checkpoint.pt'), generated_loss_weight=0.3))
    print('PREPARED', OUT, flush=True)


if __name__ == '__main__':
    main()
