"""Six-real-reference batch8 control with the existing fixed target cameras."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import vggtdiff_batch08_real_anchors as mixed

chain = mixed.chain
OUT = chain.BASE / 'vggtdiff_batch08_real6_continuous_v1'
FRAMES = (3054, 2993, 2872, 2751, 2691, 2630)


def prepare():
    chain.validate(mixed.OUT)
    baseline_manifest, target = chain.validate(mixed.BASELINE)
    OUT.mkdir(exist_ok=True)
    # The historical rectifier exposes a fixed four-frame constant. Supply six
    # in this isolated preparation process, restore it, and leave files intact.
    legacy_frames = mixed.real.bridge.anchors.FRAMES
    try:
        mixed.real.bridge.anchors.FRAMES = FRAMES
        rows = mixed.real.prepare_real(OUT)
    finally:
        mixed.real.bridge.anchors.FRAMES = legacy_frames
    assert tuple(r['original_frame'] for r in rows) == FRAMES
    assert all(rows[i]['angle'] < rows[i+1]['angle'] for i in range(5))
    sources = []
    for i, r in enumerate(rows):
        if r['original_frame'] in legacy_frames:
            old = chain.read(mixed.OUT / 'real_calibration' / f"camera_{r['original_frame']}.json")
            assert r['sha256'] == old['sha256'], 'Retained real reference pixels changed'
            np.testing.assert_array_equal(r['w2c'], old['w2c'])
            np.testing.assert_array_equal(r['intrinsics'], old['intrinsics'])
        assert r['valid_fraction'] == 1 and r['distortion_rotation_projection_check_max_px'] < 1e-3
        assert Image.open(r['image']).size == (832, 480)
        sources.append(dict(index=i, role='real', original_frame=r['original_frame'],
            original=r['image'], sha256=r['sha256'], angle=r['angle'],
            w2c=r['w2c'], intrinsics=r['intrinsics'],
            calibration=str(OUT / 'real_calibration' / f"camera_{r['original_frame']}.json")))
    trajectory = dict(source_w2c=[r['w2c'] for r in sources],
        source_intrinsics=[r['intrinsics'] for r in sources], source_angles_deg=[r['angle'] for r in sources],
        target_w2c=target['target_w2c'], target_intrinsics=target['target_intrinsics'],
        target_angles_deg=target['target_angles_deg'], orbit_indices=list(range(56,64)),
        intrinsics_image_size=[832,480], camera_convention='OpenCV W2C',
        note='Six original real images in common registered SfM/Omega world. Exact same batch8 target cameras as the two prior controls.')
    files = [Path(__file__).resolve(), Path(mixed.__file__).resolve(), Path(mixed.real.__file__).resolve(),
        Path(mixed.real.bridge.__file__).resolve(), Path(mixed.real.bridge.anchors.__file__).resolve(),
        Path(chain.__file__).resolve(), Path(mixed.previous.__file__).resolve(),
        mixed.BASELINE/'manifest.json', mixed.BASELINE/'trajectory.json',
        mixed.OUT/'manifest.json', mixed.OUT/'trajectory.json',
        mixed.real.bridge.OUT/'alignment.json', mixed.real.bridge.OUT/'pixel_correspondences.npz',
        mixed.real.bridge.ROOT/'frames_manifest.json',
        *[mixed.real.bridge.ROOT/'colmap/sparse/0'/n for n in ['images.bin','cameras.bin','points3D.bin']],
        OUT/'real_calibration/real_references.json']
    for r in rows:
        files += [Path(r['image']), Path(r['full_image']), OUT/'real_calibration'/f"camera_{r['original_frame']}.json"]
    mixed.previous.write_frozen_json(OUT/'trajectory.json', trajectory)
    files += [OUT/'trajectory.json']
    manifest = dict(batch=8, indices=list(range(56,64)), angles=target['target_angles_deg'], sources=sources,
        reference_policy='Six real video frames only; replace mixed references54/55 with real3054/2993; retain four real images and calibration exactly.',
        checkpoint=baseline_manifest['checkpoint'], omega=baseline_manifest['omega'], base=baseline_manifest['base'],
        seed=baseline_manifest['seed'], steps=baseline_manifest['steps'], resolution=[832,480],
        vram_limit_gib=20, weights=baseline_manifest['weights'],
        baseline=str(mixed.BASELINE), mixed_control=str(mixed.OUT),
        target_cameras_exactly_match_baseline=True,
        limitations='Estimated cameras, not pose GT. Replacing generated sources also changes their camera centers/poses and slightly changes source angles; not a same-source-camera pixel-only ablation.',
        frozen_files=[dict(path=str(p),sha256=chain.digest(p)) for p in dict.fromkeys(files)])
    mixed.previous.write_frozen_json(OUT/'manifest.json', manifest)
    for r in sources:
        chain.frozen_copy(Path(r['original']), OUT/'source_views'/f"{r['index']:02d}.png")
    chain.validate(OUT)
    chain.montage([OUT/'source_views'/f'{i:02d}.png' for i in range(6)],
        [f"Real {r['original_frame']} | {r['angle']:.3f} deg" for r in sources], OUT/'references6.jpg',3,(832,480))
    for p in [Path(__file__).resolve(),Path(mixed.real.__file__).resolve(),Path(chain.__file__).resolve()]:
        chain.frozen_copy(p,OUT/'implementation'/p.name)
    if not (OUT/'result.json').exists():
        chain.save(OUT/'status.json',dict(state='prepared_pending_reference_review',batch=8,real_references=6,generated_references=0))
    print('PREPARED_REAL6',OUT,flush=True)
    print('REFERENCES',[(r['original_frame'],r['angle']) for r in sources],flush=True)
    print('TARGETS',trajectory['target_angles_deg'],'seed',manifest['seed'],'steps',manifest['steps'],flush=True)
    return OUT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    output = prepare()
    if not args.prepare_only:
        review = chain.read(output/'reference_review.json')
        assert review['state']=='accepted', 'Review six real references before inference'
        chain.infer(output)
