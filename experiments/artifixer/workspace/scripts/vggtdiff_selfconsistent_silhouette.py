"""2x2 inverse-rendering/silhouette experiment; G0 reuses coverage branch B.

S0/S1 replace all 23 input images, including the six real-camera slots, with
one frozen B model's renders. This is a pipeline-consistency reference with
known artifacts, not a clean-car ground truth. G1 uses the original 23 images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

import vggtdiff_latest_coverage_ab as ab

PREVIOUS = ab.OUT
OUT = ab.BASE/'vggtdiff_selfconsistent_silhouette_v1'
SOURCE = ab.SOURCE
BRANCHES = ['S0', 'S1', 'G0', 'G1']
FACTORY = ab.training_components
INSIDE_WEIGHT = .05
OUTSIDE_WEIGHT = .2
UNCERTAINTY = 2


def validate():
    manifests = [ab.read(PREVIOUS/'experiment.json'), ab.read(OUT/'experiment.json')]
    for manifest in manifests:
        for row in manifest['frozen_files']:
            if ab.digest(Path(row['path'])) != row['sha256']:
                raise ValueError(f"Frozen input changed: {row['path']}")


def load(branch):
    import torch
    from omegaconf import OmegaConf
    import vggtdiff_car15_train
    from threedgrut.model.model import MixtureOfGaussians
    root = PREVIOUS/'B' if branch == 'G0' else OUT/branch
    result = ab.read(root/'training/result.json')
    model = MixtureOfGaussians(OmegaConf.load(root/'training/config.yaml'))
    model.init_from_checkpoint(torch.load(result['checkpoint'], map_location='cuda', weights_only=False), setup_optimizer=False)
    model.eval()
    return model


def render(model, row):
    import torch
    from threedgrut.datasets.protocols import Batch
    k = np.array(row['k'])
    blank = torch.zeros((1, 480, 832, 3), device='cuda')
    camera = torch.tensor(np.linalg.inv(row['w2c']), device='cuda', dtype=torch.float32)[None]
    batch = Batch(blank, blank, camera, intrinsics_OpenCVPinholeCameraModelParameters=dict(
        focal_length=[k[0, 0], k[1, 1]], principal_point=[k[0, 2], k[1, 2]]))
    with torch.inference_mode():
        result = model(batch, train=False)
    return (result['pred_rgb'][0].clamp(0, 1).cpu().numpy(),
            result['pred_opacity'][0, :, :, 0].cpu().numpy(),
            result['pred_dist'][0, :, :, 0].cpu().numpy())


def camera_sets():
    pool = ab.read(PREVIOUS/'pool.json')
    novel = ab.read(PREVIOUS/'B/evaluation/novel_cameras.json')
    offset = novel['offset']
    return dict(real=pool['real'], orbit=pool['generated'],
        offset=[dict(w2c=w, k=k, angle=a) for w, k, a in zip(
            offset['target_w2c'], offset['target_intrinsics'], offset['target_angles_deg'])],
        elevated=novel['elevated'], validation=ab.read(ab.OLD/'validation/cameras.json')['frames'])


def prepare():
    import torch
    from omegaconf import OmegaConf
    from plyfile import PlyData

    # Validate the existing experiment without changing any of its globals/files.
    ab.validate_frozen()
    if (OUT/'experiment.json').exists():
        raise FileExistsError('Preparation already frozen; do not overwrite it')
    torch.set_num_threads(4)
    source_result = ab.read(PREVIOUS/'B/training/result.json')
    if ab.digest(PREVIOUS/'B/model.ply') != source_result['ply_sha256']:
        raise ValueError('B model changed')
    for name in ['initial_points.ply', 'initialization.json', 'warmup_checkpoint.pt']:
        ab.frozen_copy(PREVIOUS/'shared'/name, OUT/'shared'/name)
    model = load('G0')
    rows_by_set, checks, oracle_files = camera_sets(), [], []
    pool = ab.read(PREVIOUS/'pool.json')
    for category, rows in rows_by_set.items():
        for i, row in enumerate(rows):
            rgb, alpha, depth = render(model, row)
            if not all(np.isfinite(x).all() for x in [rgb, alpha, depth]):
                raise ValueError('Non-finite oracle render')
            root = OUT/'teacher'/category
            root.mkdir(parents=True, exist_ok=True)
            path, mask_path, raw_path = root/f'{i:03d}.png', root/f'{i:03d}_mask.png', root/f'{i:03d}.npz'
            ab.write_rgb(path, rgb)
            Image.fromarray((alpha > .5).astype('uint8')*255).save(mask_path)
            np.savez_compressed(raw_path, rgb=rgb, alpha=alpha, depth=depth)
            quantized = np.asarray(Image.open(path), dtype=float)/255
            checks.append(dict(category=category, index=i, rgb_quantization_mae=float(np.abs(rgb-quantized).mean())))
            oracle_files.extend([path, mask_path, raw_path])
        print('ORACLE_RENDERED', category, len(rows), flush=True)
    # Independently reload the frozen PLY and verify that it is this exact teacher.
    from threedgrut.model.model import MixtureOfGaussians
    ply_model = MixtureOfGaussians(OmegaConf.load(PREVIOUS/'B/training/config.yaml'))
    ply_model.init_from_ply(str(PREVIOUS/'B/model.ply'), init_model=False)
    ply_model.eval()
    roundtrip = []
    for i in [0, 16, 32, 48]:
        check, _, _ = render(ply_model, rows_by_set['orbit'][i])
        with np.load(OUT/'teacher/orbit'/f'{i:03d}.npz') as reference:
            roundtrip.append(float(np.abs(check-reference['rgb']).mean()))
    assert max(roundtrip) < 1e-6
    del ply_model, model
    torch.cuda.empty_cache()
    source_rows = ab.read(PREVIOUS/'B/inputs.json')['frames']
    for branch in ['S0', 'S1', 'G1']:
        rows = []
        for row in source_rows:
            item = dict(row)
            if branch.startswith('S'):
                category = 'real' if row['kind'] == 'real' else 'orbit'
                i = int(Path(row['name']).stem) if category == 'real' else row['orbit_index']
                item.update(image=str(OUT/'teacher'/category/f'{i:03d}.png'),
                    mask=str(OUT/'teacher'/category/f'{i:03d}_mask.png'),
                    original_slot_kind=row['kind'], original_source=row.get('source'),
                    source=str(OUT/'teacher'/category/f'{i:03d}.png'),
                    supervision='Frozen B render; no original photo')
                item['sha256'] = ab.digest(Path(item['image']))
                item['mask_sha256'] = ab.digest(Path(item['mask']))
            rows.append(item)
        ab.save(OUT/branch/'inputs.json', dict(frames=rows,
            generated_indices=ab.read(PREVIOUS/'B/inputs.json')['generated_indices'],
            note='Slot kinds retain B loss weights and sampling, even when S supervision is entirely rendered.'))
    # Reuse the audited camera writer, directing only its explicit shared-cloud lookup here.
    ab.OUT = OUT
    for branch in ['S0', 'S1', 'G1']:
        ab.write_colmap(OUT/branch/'dataset', ab.read(OUT/branch/'inputs.json')['frames'])
    ab.OUT = PREVIOUS
    ab.save(OUT/'teacher/cameras.json', rows_by_set)
    ab.save(OUT/'teacher/render_audit.json', dict(ply_roundtrip_mae=roundtrip,
        reference='Known-artifact fixed B model, not clean geometry', training_image_format='8-bit RGB; binary alpha>0.5 mask',
        metric_reference='Unquantized float RGB/alpha/camera-z depth', views=checks))
    files = [Path(__file__), ab.REPO/'scripts/car15_balanced_silhouette.py',
        PREVIOUS/'B/model.ply', PREVIOUS/'B/training/result.json', Path(source_result['checkpoint']),
        PREVIOUS/'B/inputs.json', PREVIOUS/'B/training/B_sampling_plan.json',
        PREVIOUS/'B/training/config.yaml', OUT/'teacher/cameras.json']+oracle_files
    for name in ['initial_points.ply', 'initialization.json', 'warmup_checkpoint.pt']:
        files.append(OUT/'shared'/name)
    for branch in ['S0', 'S1', 'G1']:
        files.append(OUT/branch/'inputs.json')
        files.extend(sorted((OUT/branch/'dataset').rglob('*.*')))
    old_experiment = ab.read(PREVIOUS/'experiment.json')
    ab.save(OUT/'experiment.json', dict(scene_extent=old_experiment['scene_extent'], seed=ab.SEED,
        start_step=1000, final_step=5000, teacher_ply=str(PREVIOUS/'B/model.ply'),
        teacher_is_clean_ground_truth=False, G0=str(PREVIOUS/'B'),
        silhouette=dict(outside_weight=OUTSIDE_WEIGHT, inside_weight=INSIDE_WEIGHT,
                        uncertainty_pixels=UNCERTAINTY, normalizer='vehicle mask area'),
        sampling='Exact B 4000-update image-name schedule and original 1.0/0.3 slot weights',
        frozen_files=[dict(path=str(f), sha256=ab.digest(f)) for f in files]))
    paths = [Path(r['image']) for r in ab.read(OUT/'S0/inputs.json')['frames']]
    ab.montage(paths, [f"{r['name']} | {r.get('angle', 'original real pose')}" for r in source_rows],
               OUT/'oracle_training23.jpg', 3)
    ab.save(OUT/'status.json', dict(state='prepared_pending_visual_review'))
    print('PREPARED', OUT, flush=True)


def approve():
    validate()
    ab.save(OUT/'mask_review.json', dict(state='accepted',
        note='S0/S1 all 23 images are fixed-model renders. Alpha-threshold masks deliberately include teacher artifacts. '
             'These are model-consistency targets, not manually cleaned masks. G1 exactly reuses B images/masks.'))


def train(branch):
    import torch
    from car15_balanced_silhouette import balanced_silhouette
    validate()
    base = FACTORY()

    class SilhouetteTrainer(base):
        def get_losses(self, batch, outputs):
            losses = super().get_losses(batch, outputs)
            if branch in ['S1', 'G1']:
                exterior, interior = balanced_silhouette(outputs['pred_opacity'], batch.mask, UNCERTAINTY)
                slot_weight = 1. if int(Path(batch.image_name).stem) < 6 else .3
                losses['total_loss'] = losses['total_loss']-slot_weight*OUTSIDE_WEIGHT*losses['opacity_loss']
                losses['total_loss'] = losses['total_loss']+slot_weight*(OUTSIDE_WEIGHT*exterior+INSIDE_WEIGHT*interior)
                losses['opacity_loss'] = exterior
                losses['interior_loss'] = interior
            return losses

    ab.OUT = OUT
    ab.validate_frozen = validate
    ab.training_components = lambda: SilhouetteTrainer
    ab.train(branch)
    actual = ab.read(OUT/branch/'training/result.json')['actual_sample_counts']
    expected = ab.read(PREVIOUS/'B/training/result.json')['actual_sample_counts']
    assert actual == expected, 'Image exposure differs from G0'
    ab.save(OUT/branch/'sampling_audit.json', dict(exact_B_actual_counts=True, actual=actual))


def metrics(rgb, alpha, gt_rgb, gt_mask, gt_alpha=None, depth=None, gt_depth=None):
    gt_mask = np.asarray(gt_mask, dtype=bool)
    solid = alpha > .5
    tp, fp, fn = [(x).sum() for x in [solid & gt_mask, solid & ~gt_mask, ~solid & gt_mask]]
    error = rgb-gt_rgb
    result = dict(foreground_psnr=float(-10*np.log10(max(float(np.mean(error[gt_mask]**2)), 1e-12))),
        foreground_mae=float(np.abs(error[gt_mask]).mean()),
        silhouette_iou=float(tp/max(tp+fp+fn, 1)), coverage=float(tp/max(gt_mask.sum(), 1)),
        extra_area_relative_to_vehicle=float(fp/max(gt_mask.sum(), 1)),
        missing_area_relative_to_vehicle=float(fn/max(gt_mask.sum(), 1)))
    if gt_alpha is not None:
        result['full_rgb_mae'] = float(np.abs(error).mean())
        result['alpha_mae'] = float(np.abs(alpha-gt_alpha).mean())
    if gt_depth is not None:
        valid = (gt_alpha > .9) & (gt_depth > .001)
        result['relative_expected_depth_mae'] = float((np.abs(depth[valid]-gt_depth[valid])/gt_depth[valid]).mean())
    return result


def aggregate(rows):
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (float, np.floating)) and k not in ['angle']]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def evaluate(branch):
    import torch
    from plyfile import PlyData
    from omegaconf import OmegaConf
    validate()
    torch.set_num_threads(4)
    root = OUT/branch/'evaluation'
    root.mkdir(parents=True, exist_ok=True)
    model = None if branch == 'G0' else load(branch)
    sets = ab.read(OUT/'teacher/cameras.json')
    pool = ab.read(PREVIOUS/'pool.json')
    selected = ab.read(PREVIOUS/'B/inputs.json')['generated_indices']
    collected = {}
    oracle = branch.startswith('S')
    for category, camera_rows in sets.items():
        rows = []
        for i, camera in enumerate(camera_rows):
            with np.load(OUT/'teacher'/category/f'{i:03d}.npz') as data:
                reference = {k: data[k].copy() for k in data.files}
            if branch == 'G0':
                rgb, alpha, depth = reference['rgb'], reference['alpha'], reference['depth']
            else:
                rgb, alpha, depth = render(model, camera)
            ab.write_rgb(root/category/f'{i:03d}.png', rgb)
            if category in ['real', 'orbit', 'validation']:
                (root/(category+'_alpha')).mkdir(exist_ok=True)
                Image.fromarray((np.clip(alpha, 0, 1)*255).astype('uint8')).save(root/(category+'_alpha')/f'{i:03d}.png')
            score = None
            if oracle:
                score = metrics(rgb, alpha, reference['rgb'], reference['alpha'] > .5,
                                reference['alpha'], depth, reference['depth'])
            elif category in ['real', 'orbit', 'validation']:
                if category == 'validation':
                    image, mask = ab.OLD/'validation/images'/camera['name'], ab.OLD/'validation/masks'/camera['name']
                else:
                    row = pool['real' if category == 'real' else 'generated'][i]
                    image, mask = Path(row['image']), Path(row['mask'])
                score = metrics(rgb, alpha, np.asarray(Image.open(image).convert('RGB'), dtype=float)/255,
                                np.asarray(Image.open(mask).convert('L')) > 127)
            if score is not None:
                rows.append(dict(index=i, angle=camera.get('angle'), in_training=(category == 'real' or category == 'orbit' and i in selected), **score))
        collected[category] = rows
        print('EVALUATED_SET', branch, category, flush=True)
    root_model = PREVIOUS/'B' if branch == 'G0' else OUT/branch
    ply = PlyData.read(root_model/'model.ply')['vertex'].data
    assert all(np.isfinite(ply[k]).all() for k in ply.dtype.names)
    roundtrip = []
    if model is not None:
        from threedgrut.model.model import MixtureOfGaussians
        exported = MixtureOfGaussians(OmegaConf.load(root_model/'training/config.yaml'))
        exported.init_from_ply(str(root_model/'model.ply'), init_model=False)
        exported.eval()
        for i in [0, 16, 32, 48]:
            a, _, _ = render(model, sets['orbit'][i])
            b, _, _ = render(exported, sets['orbit'][i])
            roundtrip.append(float(np.abs(a-b).mean()))
        assert max(roundtrip) < 1e-6
        del exported, model
        torch.cuda.empty_cache()
    report = dict(branch=branch, oracle=oracle, gaussians=len(ply), real_or_pose6=aggregate(collected['real']),
        supervised_orbit17=aggregate([r for r in collected['orbit'] if r['in_training']]),
        heldout_orbit47=aggregate([r for r in collected['orbit'] if not r['in_training']]),
        front_rear6=aggregate([r for r in collected['orbit'] if r['index'] in ab.ADDED]),
        independent_real_or_reference_validation=aggregate(collected['validation']),
        offset64=aggregate(collected['offset']), elevated6=aggregate(collected['elevated']),
        rows=collected, ply_roundtrip_mae=roundtrip,
        caveat='S targets are exact fixed-B-model references with artifacts; G heldout generated views are not physical ground truth.')
    ab.save(root/'metrics.json', report)
    ab.save(OUT/branch/'status.json', dict(state='evaluated_pending_visual_review'))
    print('EVALUATION_COMPLETED', branch, {k: report[k] for k in ['front_rear6', 'heldout_orbit47']}, flush=True)


def compare():
    from vggtdiff_car15_keep6 import pair
    import cv2
    validate()
    for left, right in [('S0', 'S1'), ('G0', 'G1')]:
        root = OUT/f'{left}_vs_{right}'
        for category, indices in [('orbit', range(64)), ('offset', range(64)), ('elevated', range(6))]:
            for i in indices:
                angle = i*60 if category == 'elevated' else i*5.625+(2.8125 if category == 'offset' else 0)
                pair(OUT/left/'evaluation'/category/f'{i:03d}.png', OUT/right/'evaluation'/category/f'{i:03d}.png',
                     root/category/f'{i:03d}.png', f'{left} | {angle:g} deg', f'{right} | {angle:g} deg')
            if category != 'elevated':
                ab.video(root/category, root/f'{category}.mp4')
        selected = [0, 3, 8, 16, 24, 30, 32, 33, 48, 61]
        ab.montage([root/'orbit'/f'{i:03d}.png' for i in selected], [f'{i*5.625:g} deg' for i in selected],
                   root/'key_views.jpg', 2, (832, 256))
        ab.montage([root/'orbit'/f'{i:03d}.png' for i in range(64)], [f'{i*5.625:g} deg' for i in range(64)],
                   root/'all64.jpg', 4, (416, 128))
        ab.montage([root/'elevated'/f'{i:03d}.png' for i in range(6)], [f'{i*60} deg' for i in range(6)],
                   root/'elevated.jpg', 2, (832, 256))
    # Show G1 against the generated target and isolate extra/missing silhouette.
    pool = ab.read(PREVIOUS/'pool.json')['generated']
    canvas = Image.new('RGB', (1248, 268*3), '#181818')
    for j, i in enumerate([0, 32, 48]):
        gt = np.asarray(Image.open(pool[i]['mask'])) > 127
        pred = np.asarray(Image.open(OUT/'G1/evaluation/orbit_alpha'/f'{i:03d}.png')) > 127
        heat = np.zeros((480, 832, 3), dtype='uint8')
        heat[gt & pred], heat[pred & ~gt], heat[gt & ~pred] = [200, 200, 200], [255, 65, 50], [40, 130, 255]
        tiles = [(Image.open(pool[i]['image']), f'{i*5.625:g} deg target'),
                 (Image.open(OUT/'G1/evaluation/orbit'/f'{i:03d}.png'), 'G1 new silhouette'),
                 (Image.fromarray(heat), 'red extra / blue missing')]
        for k, (image, label) in enumerate(tiles):
            canvas.paste(image.resize((416, 240), Image.Resampling.LANCZOS), (k*416, j*268+28))
            ImageDraw.Draw(canvas).text((k*416+8, j*268+8), label, fill='white')
    canvas.save(OUT/'G0_vs_G1/silhouette_diagnosis.jpg', quality=95)
    m = {b: ab.read(OUT/b/'evaluation/metrics.json') for b in BRANCHES}
    rows = {b: {r['index']: r for r in m[b]['rows']['orbit']} for b in ['G0', 'G1']}
    gate = {}
    for i in [0, 32]:
        a, b = rows['G0'][i], rows['G1'][i]
        gate[str(i)] = dict(angle=i*5.625, baseline_extra=a['extra_area_relative_to_vehicle'],
            new_extra=b['extra_area_relative_to_vehicle'], baseline_coverage=a['coverage'], new_coverage=b['coverage'],
            extra_at_least_halved=b['extra_area_relative_to_vehicle'] <= .5*a['extra_area_relative_to_vehicle'],
            coverage_drop_under_one_percentage_point=b['coverage'] >= a['coverage']-.01)
    real0, real1 = [m[b]['independent_real_or_reference_validation'] for b in ['G0', 'G1']]
    gate['independent_real'] = dict(psnr_delta=real1['foreground_psnr']-real0['foreground_psnr'],
                                  iou_delta=real1['silhouette_iou']-real0['silhouette_iou'])
    ab.save(OUT/'acceptance_gate.json', gate)
    expected = ab.read(PREVIOUS/'B/training/result.json')['actual_sample_counts']
    for branch in ['S0', 'S1', 'G1']:
        assert ab.read(OUT/branch/'training/result.json')['actual_sample_counts'] == expected
    videos = []
    for path in sorted(OUT.glob('*_vs_*/*.mp4')):
        cap = cv2.VideoCapture(str(path)); count = 0; fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok: break
            assert frame.shape == (512, 1664, 3)
            count += 1
        cap.release()
        assert count == 64 and fps == 8
        videos.append(dict(path=str(path), frames=count, fps=fps))
    ab.save(OUT/'audit.json', dict(frozen_files_verified=True, exact_G0_sampling=True,
        all_new_steps=5000, videos=videos,
        ply_roundtrip_mae={b: m[b]['ply_roundtrip_mae'] for b in ['S0', 'S1', 'G1']}))
    lines = ['# 自洽监督 × 轮廓约束实验', '',
        'S0/S1：全部二十三张图来自固定 B 模型的渲染；G0 为原 B，G1 使用完全相同的真实/生成监督。',
        'S1/G1 仅替换轮廓损失：掩码外 alpha 按车辆面积归一化，内部覆盖项权重 .05，边界不确定带 2 像素。',
        'RGB、SSIM、形状正则、相机、初始化、1000→5000 步和逐图抽样均不变。S 的原真实相机槽位仍使用原采样/权重，但图像已经全部换成渲染。', '',
        '## 自洽参考重建', '', '| 指标 | S0 | S1 |', '|---|---:|---:|']
    for category in ['supervised_orbit17', 'heldout_orbit47', 'offset64', 'elevated6']:
        for key in ['foreground_psnr', 'silhouette_iou', 'alpha_mae', 'relative_expected_depth_mae']:
            lines.append(f"| {category} / {key} | {m['S0'][category][key]:.6f} | {m['S1'][category][key]:.6f} |")
    lines += ['', '## 真实/生成监督', '', '| 指标 | G0 | G1 |', '|---|---:|---:|']
    for category in ['front_rear6', 'heldout_orbit47', 'independent_real_or_reference_validation']:
        for key in ['foreground_psnr', 'silhouette_iou', 'coverage', 'extra_area_relative_to_vehicle']:
            lines.append(f"| {category} / {key} | {m['G0'][category][key]:.6f} | {m['G1'][category][key]:.6f} |")
    lines += ['', '## 解释边界', '',
        '固定 B 模型本身含拉丝和外壳。S 实验只测相对于该已知模型的逆渲染一致性，不评价干净车体恢复；不能仅凭 S 指标优于 G 就证明所有差异都来自生成不一致。',
        'Oracle RGB 训练图保存为八位 PNG，原始浮点 RGB/alpha/期望深度另存 npz；训练掩码由 alpha>0.5 得到，弱透明部分的阈值化以及形状/轮廓正则意味着不应期待总损失为零。',
        'G 的生成图验证仍是合成一致性，真实独立验证只有两帧且含 PnP 位姿误差。单种子结果不是统计显著性结论。', '',
        '## 产物', '', '- S0_vs_S1/orbit.mp4、offset.mp4：自洽监督对照。',
        '- G0_vs_G1/orbit.mp4、offset.mp4：真实/生成监督对照。',
        '- 各对照的 all64.jpg、key_views.jpg、elevated.jpg。',
        '- S0、S1、G1/model.ply 和 3000/5000 步检查点。G0 模型仍在原 coverage 实验 B/model.ply。',
        '- acceptance_gate.json：预先约定的前后多余轮廓/覆盖率检查。',
        '- audit.json：输入、采样、PLY 回读和视频核验。', '',
        '复现：.venv-car15/bin/python scripts/vggtdiff_selfconsistent_silhouette.py，依次 prepare、检查监督后 approve、train --branch S0/S1/G1、evaluate --branch S0/S1/G0/G1、compare。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    ab.save(OUT/'status.json', dict(state='compared_pending_visual_review', branches=BRANCHES))
    print('COMPARISON_COMPLETED', gate, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'approve', 'train', 'evaluate', 'compare'])
    parser.add_argument('--branch', choices=BRANCHES)
    args = parser.parse_args()
    if args.action in ['train', 'evaluate']:
        if args.branch is None or args.action == 'train' and args.branch == 'G0':
            parser.error('Choose a valid branch; G0 reuses the existing model')
        globals()[args.action](args.branch)
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
