"""Compare the elongated-Gaussian penalty against the group-A reconstruction."""
import math
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from plyfile import PlyData

import vggtdiff_realvideo_unseen as e
from vggtdiff_car15_groupa_prepare import OLD, OUT as GROUP
from vggtdiff_car15_groupa_tight_train import OUT
from vggtdiff_car15_prepare import BASE
from vggtdiff_white_car import digest, save
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch


def read(path):
    import json
    return json.loads(path.read_text())


def load():
    result = read(OUT / 'training/tight_result.json')
    conf = OmegaConf.load(OUT / 'training/tight_config.yaml')
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
    return dict(foreground_psnr=-10 * math.log10(max(mse, 1e-12)), silhouette_iou=float((solid & mask).sum() / max((solid | mask).sum(), 1)))


def pair(left, right, path, left_label, right_label):
    image = Image.new('RGB', (1664, 512))
    image.paste(Image.open(left), (0, 32))
    image.paste(Image.open(right), (832, 32))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), left_label, fill='white')
    draw.text((842, 10), right_label, fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    torch.set_num_threads(4)
    root = OUT / 'evaluation'
    model = load()
    rows = []
    for frame in read(GROUP / 'inputs.json')['frames']:
        rgb, alpha = render(model, np.array(frame['w2c']), frame['k'])
        ground_truth = np.array(Image.open(GROUP / 'images' / frame['name'])) / 255.
        mask = np.array(Image.open(GROUP / 'masks' / frame['name'])) > 127
        rows.append(dict(name=frame['name'], kind=frame['kind'], angle=frame.get('angle'), **metrics(rgb, alpha, ground_truth, mask)))
    validation = read(OLD / 'validation/cameras.json')
    if isinstance(validation, dict):
        validation = validation.get('frames', validation.get('cameras', validation))
    validation_rows = []
    for frame in validation:
        name = Path(frame['name']).name
        rgb, alpha = render(model, np.array(frame['w2c']), frame.get('k', frame.get('intrinsics')))
        ground_truth = np.array(Image.open(OLD / 'validation/images' / name)) / 255.
        mask = np.array(Image.open(OLD / 'validation/masks' / name)) > 127
        validation_rows.append(dict(name=name, **metrics(rgb, alpha, ground_truth, mask)))
    orbit = read(BASE / 'vggtdiff_realvideo_orbit360_64_v2_batch8/trajectory.json')
    for index, (w2c, intrinsics) in enumerate(zip(orbit['target_w2c'], orbit['target_intrinsics'])):
        rgb, alpha = render(model, np.array(w2c), intrinsics)
        write(root / 'orbit' / f'{index:03d}.png', rgb)
    del model
    torch.cuda.empty_cache()
    for index in range(64):
        angle = index * 360 / 64
        pair(GROUP / 'evaluation/orbit' / f'{index:03d}.png', root / 'orbit' / f'{index:03d}.png', root / 'versus_groupa' / f'{index:03d}.png', f'group A | {angle:.3f} deg', f'tight | {angle:.3f} deg')
    subprocess.run(['ffmpeg', '-y', '-framerate', '8', '-i', str(root / 'versus_groupa/%03d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(root / 'orbit_versus_groupa.mp4')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    selected = [8, 16, 20, 24, 32]
    e.contact([root / 'versus_groupa' / f'{index:03d}.png' for index in selected], [f'{index * 5.625:.2f} deg' for index in selected], root / 'right_side_contact.jpg', 1)
    ply = PlyData.read(OUT / 'tight.ply')['vertex']
    scales = np.exp(np.stack([ply['scale_0'], ply['scale_1'], ply['scale_2']], 1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-ply['opacity'].astype(np.float64)))
    ratio = scales.max(1) / np.clip(scales.min(1), 1e-12, None)

    def mean(items, key):
        return float(np.mean([item[key] for item in items]))

    real = [row for row in rows if row['kind'] == 'real']
    generated = [row for row in rows if row['kind'] == 'generated']
    report = dict(gaussians=len(ply), elongated_opacity_mass_ratio16=float(opacity[ratio > 16].sum() / opacity.sum()),
                  real_training=dict(foreground_psnr=mean(real, 'foreground_psnr'), silhouette_iou=mean(real, 'silhouette_iou')),
                  generated_training=dict(foreground_psnr=mean(generated, 'foreground_psnr'), silhouette_iou=mean(generated, 'silhouette_iou')),
                  independent_real=dict(foreground_psnr=mean(validation_rows, 'foreground_psnr'), silhouette_iou=mean(validation_rows, 'silhouette_iou')))
    save(root / 'metrics.json', report)
    print('EVALUATION_COMPLETED', report, 'ply', digest(OUT / 'tight.ply'), flush=True)


if __name__ == '__main__':
    main()
