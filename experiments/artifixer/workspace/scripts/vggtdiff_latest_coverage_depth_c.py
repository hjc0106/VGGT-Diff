"""Conditional C experiment: B's exact data/sampling plus real-only depth anchors.

Depth is an estimated prior, not ground truth. Only high-confidence real-image
points supported by at least two other real views are admitted. No synthetic
depth, independently estimated scale, or validation images enter the loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

import vggtdiff_latest_coverage_ab as ab

ROOT = ab.OUT/'C'
DENSE = ab.BASE/'vggtdiff_realvideo_unseen_right_v1/omega_dense.npz'
DEPTH_WEIGHT = .05
RELATIVE_TOLERANCE = .01
MAX_NEAREST_PIXEL_DISTANCE = 3.
MIN_OTHER_VIEWS = 2


def project(points, row):
    w, k = np.array(row['w2c']), np.array(row['k'])
    camera = points@w[:3, :3].T+w[:3, 3]
    image = camera@k.T
    uv = image[:, :2]/np.maximum(image[:, 2:3], 1e-12)
    xy = np.rint(np.nan_to_num(uv, nan=-10000, posinf=10000, neginf=-10000)).astype(int)
    valid = np.isfinite(points).all(1) & np.isfinite(uv).all(1) & (camera[:, 2] > .001)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < 832) & (xy[:, 1] >= 0) & (xy[:, 1] < 480)
    return xy, camera[:, 2], valid


def prepare():
    ab.validate_frozen()
    rows = ab.read(ab.OUT/'pool.json')['real']
    masks = [binary_erosion(np.asarray(Image.open(row['mask']).convert('L')) > 127,
                            structure=np.ones((3, 3)), iterations=2) for row in rows]
    with np.load(DENSE) as data:
        world, confidence = data['world'].copy(), data['confidence'].copy()
    own_points, maps, audit = [], [], []
    for i, row in enumerate(rows):
        points, conf = world[i].reshape(-1, 3), confidence[i].reshape(-1)
        xy, z, valid = project(points, row)
        ids = np.flatnonzero(valid)
        ids = ids[masks[i][xy[ids, 1], xy[ids, 0]]]
        cutoff = float(np.quantile(conf[ids], .6))
        ids = ids[np.isfinite(conf[ids]) & (conf[ids] >= cutoff)]
        # One frontmost high-confidence point per projected pixel.
        ids = ids[np.argsort(z[ids], kind='stable')]
        pixels = xy[ids, 1]*832+xy[ids, 0]
        _, first = np.unique(pixels, return_index=True)
        ids = ids[first]
        depth = np.zeros((480, 832), dtype=np.float32)
        depth[xy[ids, 1], xy[ids, 0]] = z[ids]
        distance, nearest = distance_transform_edt(depth == 0, return_indices=True)
        maps.append((depth, depth[nearest[0], nearest[1]], distance))
        own_points.append((points[ids], xy[ids], z[ids]))
        audit.append(dict(real_index=i, confidence_quantile=.6, confidence_cutoff=cutoff,
                          high_confidence_unique_pixels=len(ids)))
    (ROOT/'depth_prior').mkdir(parents=True, exist_ok=True)
    for i, (points, own_xy, own_z) in enumerate(own_points):
        support = np.zeros(len(points), dtype=int)
        consistency_errors = []
        for j, row in enumerate(rows):
            if i == j:
                continue
            xy, z, valid = project(points, row)
            ids = np.flatnonzero(valid)
            ids = ids[masks[j][xy[ids, 1], xy[ids, 0]]]
            target = maps[j][1][xy[ids, 1], xy[ids, 0]]
            nearby = maps[j][2][xy[ids, 1], xy[ids, 0]] <= MAX_NEAREST_PIXEL_DISTANCE
            relative_error = np.abs(z[ids]-target)/np.maximum(target, 1e-6)
            supported = nearby & (target > 0) & (relative_error <= RELATIVE_TOLERANCE)
            support[ids[supported]] += 1
            consistency_errors.extend(relative_error[supported].tolist())
        keep = support >= MIN_OTHER_VIEWS
        depth = np.zeros((480, 832), dtype=np.float32)
        depth[own_xy[keep, 1], own_xy[keep, 0]] = own_z[keep]
        consistent_count = int(keep.sum())
        # Keep the agreement threshold strict; do not relax it to fill an unsupported view.
        if consistent_count < 200:
            keep[:] = False
            depth[:] = 0
        np.savez_compressed(ROOT/'depth_prior'/f'{i:02d}.npz', depth=depth, valid=depth > 0)
        image = np.asarray(Image.open(rows[i]['image']).convert('RGB')).copy()
        image[depth > 0] = (.35*image[depth > 0]+.65*np.array([255, 80, 0])).astype('uint8')
        Image.fromarray(image).save(ROOT/'depth_prior'/f'{i:02d}_support.png')
        audit[i].update(accepted_pixels=int(keep.sum()), consistent_candidates=consistent_count,
                        fraction_of_vehicle=float(keep.sum()/masks[i].sum()),
                        median_agreement_error=float(np.median(consistency_errors)) if consistency_errors else None,
                        depth_minmax=[float(own_z[keep].min()), float(own_z[keep].max())] if keep.any() else None,
                        excluded_reason='Fewer than 200 cross-view supported pixels' if not keep.any() else None)
    if sum(a['accepted_pixels'] for a in audit) < 2000:
        raise ValueError('Insufficient overall depth support; do not train C')
    ab.montage([ROOT/'depth_prior'/f'{i:02d}_support.png' for i in range(6)],
               [f"real {i:02d} | {audit[i]['accepted_pixels']} anchors" for i in range(6)], ROOT/'depth_support.jpg')
    # Same images, masks, cameras, initialization and schedule as B.
    if not (ROOT/'dataset').exists():
        shutil.copytree(ab.OUT/'B/dataset', ROOT/'dataset')
    ab.frozen_copy(ab.OUT/'B/inputs.json', ROOT/'inputs.json')
    frozen = [Path(__file__), DENSE]+[ROOT/'depth_prior'/f'{i:02d}.npz' for i in range(6)]
    ab.save(ROOT/'depth_prior/manifest.json', dict(
        real_only=True, geometry_coordinates='Existing Omega world; project with each real camera, no scale fitting',
        source=str(DENSE), relative_depth_tolerance=RELATIVE_TOLERANCE,
        nearest_support_distance_pixels=MAX_NEAREST_PIXEL_DISTANCE, min_other_views=MIN_OTHER_VIEWS,
        depth_loss_weight=DEPTH_WEIGHT, loss='SmoothL1(log(rendered_camera_z)-log(prior_camera_z)), beta=.02',
        anchors=audit, frozen_files=[dict(path=str(p), sha256=ab.digest(p)) for p in frozen],
        caveat='Cross-view agreement of an estimated prior is not independent evidence of geometric accuracy.'))
    ab.save(ROOT/'status.json', dict(state='depth_prepared_pending_review'))
    print('DEPTH_PREPARED', audit, flush=True)


def validate():
    ab.validate_frozen()
    for row in ab.read(ROOT/'depth_prior/manifest.json')['frozen_files']:
        if ab.digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f"Frozen depth input changed: {row['path']}")


def train():
    import torch
    import torch.nn.functional as F
    validate()
    base = ab.training_components()

    class DepthTrainer(base):
        def get_losses(self, batch, outputs):
            losses = super().get_losses(batch, outputs)
            stem = int(Path(batch.image_name).stem)
            depth_loss = outputs['pred_dist'].new_zeros(())
            if stem < 6:
                if not hasattr(self, 'depth_cache'):
                    self.depth_cache = {}
                if stem not in self.depth_cache:
                    with np.load(ROOT/'depth_prior'/f'{stem:02d}.npz') as data:
                        self.depth_cache[stem] = (
                            torch.from_numpy(data['depth']).to(self.device)[None, ..., None],
                            torch.from_numpy(data['valid']).to(self.device)[None, ..., None])
                reference, valid = self.depth_cache[stem]
                prediction = outputs['pred_dist']
                if bool(valid.any()):
                    residual = torch.log(prediction[valid].clamp_min(1e-5))-torch.log(reference[valid])
                    depth_loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.02)
            losses['total_loss'] = losses['total_loss']+DEPTH_WEIGHT*depth_loss
            losses['depth_loss'] = depth_loss
            return losses

    # Reuse the exact audited A/B runner without altering its frozen source.
    ab.training_components = lambda: DepthTrainer
    ab.train('C')
    ab.save(ROOT/'depth_training.json', dict(weight=DEPTH_WEIGHT, real_images_only=True,
        valid_pixels=[r['accepted_pixels'] for r in ab.read(ROOT/'depth_prior/manifest.json')['anchors']]))


def compare():
    from vggtdiff_car15_keep6 import pair
    validate()
    root = ab.OUT/'comparison_C'
    for category, indices in [('orbit', range(64)), ('offset', range(64)), ('elevated', range(0, 360, 60))]:
        for i in indices:
            angle = i if category == 'elevated' else i*5.625+(2.8125 if category == 'offset' else 0)
            pair(ab.OUT/'B/evaluation'/category/f'{i:03d}.png', ROOT/'evaluation'/category/f'{i:03d}.png',
                 root/category/f'{i:03d}.png', f'B coverage | {angle:g} deg', f'C plus real depth | {angle:g} deg')
        if category != 'elevated':
            ab.video(root/category, root/f'{category}_B_vs_C.mp4')
    selected = [0, 3, 8, 16, 24, 30, 32, 33, 48, 61]
    ab.montage([root/'orbit'/f'{i:03d}.png' for i in selected], [f'{i*5.625:g} deg' for i in selected],
               root/'key_views.jpg', 2, (832, 256))
    ab.montage([root/'orbit'/f'{i:03d}.png' for i in range(64)], [f'{i*5.625:g} deg' for i in range(64)],
               root/'all64.jpg', 4, (416, 128))
    ab.montage([root/'elevated'/f'{i:03d}.png' for i in range(0, 360, 60)], [f'{i} deg' for i in range(0, 360, 60)],
               root/'elevated.jpg', 2, (832, 256))
    b, c = [ab.read(ab.OUT/branch/'training/result.json') for branch in ['B', 'C']]
    assert b['actual_sample_counts'] == c['actual_sample_counts']
    assert b['warmup_sha256'] == c['warmup_sha256']
    ab.save(ROOT/'comparison_audit.json', dict(same_actual_sampling=True, same_warmup=True,
        only_loss_addition='Estimated real-image depth anchors, weight .05', state='pending_visual_review'))
    print('C_COMPARISON_COMPLETED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate', 'compare'])
    args = parser.parse_args()
    if args.action == 'evaluate':
        validate()
        ab.evaluate('C')
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
