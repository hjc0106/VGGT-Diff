"""Batch8 controlled real-anchor ablation: four real + two batch7 generated.

Targets, checkpoint, seed and steps match the prior all-generated-reference run.
--prepare-only verifies and freezes inputs without loading diffusion models.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import vggtdiff_real_front_references as real
import vggtdiff_batch07_continuous as previous

chain=previous.chain
OUT=real.OUT
BASELINE=chain.BASE/'vggtdiff_batch06_07gen_to_batch08_continuous_v1'
B7=previous.DEFAULT_OUTPUT


def prepare(args):
    output=args.output.resolve()
    if output!=OUT.resolve():
        raise ValueError('This calibrated experiment uses its fixed output directory; use a new script/config for another calibration.')
    if args.steps<=0 or args.vram_limit_gib<0:raise ValueError('Invalid steps/VRAM limit')
    chain.validate(BASELINE);chain.validate(B7)
    baseline=chain.read(BASELINE/'trajectory.json');baseline_manifest=chain.read(BASELINE/'manifest.json')
    rows=real.prepare_real(output)
    gen_t=chain.read(B7/'trajectory.json');gen_r=chain.read(B7/'result.json')
    if gen_r['cameras_sha256']!=chain.digest(B7/'trajectory.json'):raise ValueError('Generated reference camera mismatch')
    sources=[]
    for local in [6,7]:
        r=gen_r['frames'][local];path=Path(r['path'])
        if chain.digest(path)!=r['sha256']:raise ValueError('Generated reference changed')
        sources.append(dict(index=len(sources),role='generated',source_batch=7,source_local_index=local,
            original=str(path),sha256=r['sha256'],angle=gen_t['target_angles_deg'][local],
            w2c=gen_t['target_w2c'][local],intrinsics=gen_t['target_intrinsics'][local]))
    for r in rows:
        sources.append(dict(index=len(sources),role='real',original_frame=r['original_frame'],
            original=r['image'],sha256=r['sha256'],angle=r['angle'],w2c=r['w2c'],intrinsics=r['intrinsics'],
            calibration=str(output/'real_calibration'/f"camera_{r['original_frame']}.json")))
    if [s['role'] for s in sources]!=['generated']*2+['real']*4:raise ValueError('Unexpected reference composition')
    if not all(sources[i]['angle']<sources[i+1]['angle'] for i in range(5)):raise ValueError('References must follow angular order')
    for r in sources:
        if chain.digest(Path(r['original']))!=r['sha256']:raise ValueError('Reference hash mismatch')
        with Image.open(r['original']) as image:
            if image.size!=(832,480):raise ValueError('Invalid reference resolution')
    trajectory=dict(source_w2c=[r['w2c'] for r in sources],source_intrinsics=[r['intrinsics'] for r in sources],
        source_angles_deg=[r['angle'] for r in sources],target_w2c=baseline['target_w2c'],target_intrinsics=baseline['target_intrinsics'],
        target_angles_deg=baseline['target_angles_deg'],orbit_indices=list(range(56,64)),intrinsics_image_size=[832,480],camera_convention='OpenCV W2C',
        note='Four real front views registered by full-videoSfM/background-pixel Sim3, plus generated batch7 local6/7. Exact same target cameras as all-generated batch8 baseline.')
    np.testing.assert_array_equal(trajectory['target_w2c'],baseline['target_w2c'])
    np.testing.assert_array_equal(trajectory['target_intrinsics'],baseline['target_intrinsics'])
    assert np.asarray(trajectory['source_w2c']+trajectory['target_w2c']).shape==(14,4,4)
    assert np.asarray(trajectory['source_intrinsics']+trajectory['target_intrinsics']).shape==(14,3,3)
    files=[Path(__file__).resolve(),Path(real.__file__).resolve(),Path(real.bridge.__file__).resolve(),Path(real.bridge.anchors.__file__).resolve(),
        Path(chain.__file__).resolve(),Path(previous.__file__).resolve(),Path(real.bridge.anchors.gt.__file__).resolve(),
        BASELINE/'manifest.json',BASELINE/'trajectory.json',B7/'result.json',B7/'trajectory.json',
        real.bridge.ROOT/'frames_manifest.json',real.bridge.ROOT/'colmap/sparse/0/images.bin',real.bridge.ROOT/'colmap/sparse/0/cameras.bin',
        real.bridge.ROOT/'colmap/sparse/0/points3D.bin',real.bridge.OUT/'alignment.json',real.bridge.OUT/'pixel_correspondences.npz',
        real.bridge.anchors.CACHE,real.bridge.anchors.gt.REAL/'camera_estimate.json',real.bridge.anchors.gt.REAL/'geometry_check.json',
        output/'real_calibration/real_references.json']
    files += [Path(r['original']) for r in sources]
    for r in rows:files += [Path(r['full_image']),output/'real_calibration'/f"camera_{r['original_frame']}.json"]
    for w in baseline_manifest['weights']:
        stat=Path(w['path']).stat()
        if [stat.st_size,stat.st_mtime_ns]!=[w['size'],w['mtime_ns']]:raise ValueError('Baseline checkpoint weights changed')
    manifest=dict(batch=8,indices=list(range(56,64)),angles=trajectory['target_angles_deg'],sources=sources,
        reference_policy='Four real frames2872/2751/2691/2630 + two generated batch7 views54/55. No generated pixels in real references.',
        checkpoint=baseline_manifest['checkpoint'],omega=baseline_manifest['omega'],base=baseline_manifest['base'],
        seed=args.seed,steps=args.steps,resolution=[832,480],vram_limit_gib=args.vram_limit_gib,weights=baseline_manifest['weights'],
        baseline=str(BASELINE),target_cameras_exactly_match_baseline=True,
        seed_matches_baseline=args.seed==baseline_manifest['seed'],steps_match_baseline=args.steps==baseline_manifest['steps'],
        frozen_files=[dict(path=str(p),sha256=chain.digest(p)) for p in dict.fromkeys(files)])
    audit=dict(state='passed',real_reference_frames=[r['original_frame'] for r in rows],
        generated_reference_indices=[54,55],target_indices=list(range(56,64)),
        same_target_poses_and_K_as_baseline=True,source_camera_centers_unchanged_by_rectification=True,
        real_pixels_only=True,distortion_rotation_projection_max_px=max(r['distortion_rotation_projection_check_max_px'] for r in rows),
        all_reference_pixels_supported=all(r['valid_fraction']==1 for r in rows),
        alignment=real.bridge.anchors.gt.read(real.bridge.OUT/'alignment.json'),
        limitations='Estimated SfM/Omega alignment with heldout background residual; not pose GT. This experiment changes reference composition, cameras and angular span, not just image realism at identical source poses.')
    artifacts=[('trajectory.json',trajectory),('preparation_audit.json',audit)]
    for name,value in artifacts:
        manifest['frozen_files'].append(dict(path=str(output/name),sha256=hashlib.sha256((json.dumps(value,indent=2)+'\n').encode()).hexdigest()))
    for name,value in artifacts+[('manifest.json',manifest)]:
        if (output/name).exists() and chain.read(output/name)!=value:raise ValueError(f'Frozen experiment differs: {output/name}')
    for name,value in artifacts+[('manifest.json',manifest)]:previous.write_frozen_json(output/name,value)
    for r in sources:chain.frozen_copy(Path(r['original']),output/'source_views'/f"{r['index']:02d}.png")
    chain.validate(output)
    chain.montage([output/'source_views'/f'{i:02d}.png' for i in range(6)],
        [f"Ref{i} | {r['role']} | {r.get('original_frame',r.get('source_local_index'))} | {r['angle']:.2f}deg" for i,r in enumerate(sources)],
        output/'references6.jpg',3,(832,480))
    for p in [Path(__file__).resolve(),Path(real.__file__).resolve(),Path(real.bridge.__file__).resolve(),Path(chain.__file__).resolve()]:
        chain.frozen_copy(p,output/'implementation'/p.name)
    if not (output/'result.json').exists():chain.save(output/'status.json',dict(state='prepared',batch=8,real_references=4,generated_references=2))
    print('PREPARED_REAL4_GEN2',output,flush=True)
    print('REFERENCES',[(r['role'],r.get('original_frame',r.get('source_local_index')),r['angle']) for r in sources],flush=True)
    print('TARGETS_UNCHANGED',trajectory['target_angles_deg'],'seed',args.seed,'steps',args.steps,flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--vram-limit-gib',type=float,default=20)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();output=prepare(args)
    if not args.prepare_only:chain.infer(output)

if __name__=='__main__':main()
