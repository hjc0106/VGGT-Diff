"""Render the group-A model and compare it with the frozen real6 and mixed15 orbits."""
import math
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from plyfile import PlyData

import vggtdiff_car15_train as train
import vggtdiff_realvideo_unseen as e
from vggtdiff_car15_groupa_prepare import OLD, OUT
from vggtdiff_car15_prepare import BASE
from vggtdiff_white_car import digest, save
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch


def read(path):
    import json
    return json.loads(path.read_text())


def load():
    result = read(OUT / 'training/groupa_result.json')
    conf = OmegaConf.load(OUT / 'training/groupa_config.yaml')
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(torch.load(result['checkpoint'], map_location='cuda', weights_only=False), setup_optimizer=False)
    model.eval()
    return model


@torch.inference_mode()
def render(model, w2c, intrinsics):
    intrinsics = np.array(intrinsics)
    blank = torch.zeros((1, 480, 832, 3), device='cuda')
    batch = Batch(blank, blank, torch.tensor(np.linalg.inv(w2c), dtype=torch.float32, device='cuda')[None], intrinsics_OpenCVPinholeCameraModelParameters=dict(focal_length=[intrinsics[0, 0], intrinsics[1, 1]], principal_point=[intrinsics[0, 2], intrinsics[1, 2]]))
    output = model(batch, train=False)
    return output['pred_rgb'][0].clamp(0, 1).cpu().numpy(), output['pred_opacity'][0, :, :, 0].cpu().numpy()


def write(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(path)


def metrics(rgb, alpha, ground_truth, mask):
    error = (rgb - ground_truth)[mask]
    mse = float(np.mean(error ** 2))
    solid = alpha > .5
    return dict(foreground_psnr=-10 * math.log10(max(mse, 1e-12)), foreground_mae=float(abs(error).mean()), silhouette_iou=float((solid & mask).sum() / max((solid | mask).sum(), 1)), foreground_mean_opacity=float(alpha[mask].mean()))


def pair(left, right, path, left_label, right_label):
    image = Image.new('RGB', (1664, 512))
    image.paste(Image.open(left), (0, 32))
    image.paste(Image.open(right), (832, 32))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), left_label, fill='white')
    draw.text((842, 10), right_label, fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def encode(pattern, destination):
    subprocess.run(['ffmpeg', '-y', '-framerate', '8', '-i', str(pattern), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(destination)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    torch.set_num_threads(4)
    root = OUT / 'evaluation'
    root.mkdir(exist_ok=True)
    model = load()
    rows = []
    for frame in read(OUT / 'inputs.json')['frames']:
        rgb, alpha = render(model, np.array(frame['w2c']), frame['k'])
        write(root / 'training' / frame['name'], rgb)
        ground_truth = np.array(Image.open(OUT / 'images' / frame['name'])) / 255.
        mask = np.array(Image.open(OUT / 'masks' / frame['name'])) > 127
        rows.append(dict(name=frame['name'], kind=frame['kind'], angle=frame.get('angle'), **metrics(rgb, alpha, ground_truth, mask)))
    validation_rows = []
    validation = read(OLD / 'validation/cameras.json')
    if isinstance(validation, dict):
        validation = validation.get('frames', validation.get('cameras', validation))
    for frame in validation:
        name = Path(frame['name']).name
        rgb, alpha = render(model, np.array(frame['w2c']), frame.get('k', frame.get('intrinsics')))
        write(root / 'validation' / name, rgb)
        ground_truth = np.array(Image.open(OLD / 'validation/images' / name)) / 255.
        mask = np.array(Image.open(OLD / 'validation/masks' / name)) > 127
        validation_rows.append(dict(name=name, **metrics(rgb, alpha, ground_truth, mask)))
    orbit = read(BASE / 'vggtdiff_realvideo_orbit360_64_v2_batch8/trajectory.json')
    for index, (w2c, intrinsics) in enumerate(zip(orbit['target_w2c'], orbit['target_intrinsics'])):
        rgb, alpha = render(model, np.array(w2c), intrinsics)
        write(root / 'orbit' / f'{index:03d}.png', rgb)
    geometry = read(BASE / 'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json')
    center = np.array(geometry['center'])
    up = np.array(geometry['up'])
    front = np.array(geometry['front'])
    left = np.array(geometry['left'])
    for angle in range(0, 360, 60):
        theta = np.deg2rad(angle)
        position = center + .3142 * (np.cos(theta) * front - np.sin(theta) * left) + .10749 * up
        rgb, alpha = render(model, e.look_at(position, center, up), orbit['target_intrinsics'][0])
        write(root / 'elevated' / f'{angle:03d}.png', rgb)
    del model
    torch.cuda.empty_cache()
    previous = OLD / 'evaluation'
    for index in range(64):
        angle = index * 360 / 64
        pair(previous / 'real6/orbit' / f'{index:03d}.png', root / 'orbit' / f'{index:03d}.png', root / 'versus_real6' / f'{index:03d}.png', f'real6 | {angle:.3f} deg', f'group A | {angle:.3f} deg')
        pair(previous / 'mixed15/orbit' / f'{index:03d}.png', root / 'orbit' / f'{index:03d}.png', root / 'versus_mixed15' / f'{index:03d}.png', f'mixed15 | {angle:.3f} deg', f'group A | {angle:.3f} deg')
    encode(root / 'versus_real6/%03d.png', root / 'orbit_versus_real6.mp4')
    encode(root / 'versus_mixed15/%03d.png', root / 'orbit_versus_mixed15.mp4')
    selected = [8, 16, 18, 20, 24, 32]
    e.contact([root / 'versus_mixed15' / f'{index:03d}.png' for index in selected], [f'{index * 5.625:.2f} deg' for index in selected], root / 'right_side_contact.jpg', 2)
    ply = PlyData.read(OUT / 'groupa.ply')['vertex']
    assert all(np.isfinite(ply[name]).all() for name in ply.data.dtype.names)
    scales = np.exp(np.stack([ply['scale_0'], ply['scale_1'], ply['scale_2']], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-ply['opacity'].astype(np.float64)))
    ratio = scales.max(1) / np.clip(scales.min(1), 1e-12, None)

    def mean(items, key):
        return float(np.mean([item[key] for item in items]))

    generated = [row for row in rows if row['kind'] == 'generated']
    real = [row for row in rows if row['kind'] == 'real']
    report = dict(gaussians=len(ply), elongated_opacity_mass_ratio16=float(opacity[ratio > 16].sum() / opacity.sum()),
                  real_training=dict(foreground_psnr=mean(real, 'foreground_psnr'), silhouette_iou=mean(real, 'silhouette_iou')),
                  generated_training=dict(foreground_psnr=mean(generated, 'foreground_psnr'), silhouette_iou=mean(generated, 'silhouette_iou')),
                  independent_real=dict(foreground_psnr=mean(validation_rows, 'foreground_psnr'), silhouette_iou=mean(validation_rows, 'silhouette_iou')),
                  training_metrics=rows, independent_real_metrics=validation_rows, ply_sha256=digest(OUT / 'groupa.ply'))
    save(root / 'metrics.json', report)
    print('EVALUATION_COMPLETED', report['gaussians'], report['real_training'], report['generated_training'], report['independent_real'], report['elongated_opacity_mass_ratio16'], flush=True)


if __name__ == '__main__':
    main()
