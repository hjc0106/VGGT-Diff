"""Regenerate batch7 with three generated references from each neighbor batch.

Batch6 is seed20261028; batch8 is the six-real-conditioned generated control.
Reuse batch7 target cameras exactly; do not load diffusion with --prepare-only.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import vggtdiff_batch07_continuous as previous
import vggtdiff_batch08_real6_continuous as real8

chain=previous.chain
B6=chain.BASE/'crossview_geometry_matched_gt_v1/seed_20261028'
B8=real8.OUT
OLD7=previous.DEFAULT_OUTPUT
OUT=chain.BASE/'vggtdiff_batch06_08gen_to_batch07_continuous_v1'
SELECTION=((6,5),(6,6),(6,7),(8,0),(8,1),(8,2))


def prepare(args):
    if args.seed<0 or args.steps<=0 or args.vram_limit_gib<0:
        raise ValueError('Invalid seed, sampling steps or VRAM limit')
    output=args.output.resolve()
    protected=[B6.resolve(),B8.resolve(),OLD7.resolve()]
    if output==chain.BASE.resolve() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be independent of all source runs')
    old_manifest,old_target=chain.validate(OLD7)
    chain.validate(B8)
    protocol=chain.read(B6.parent/'protocol.json')
    assert protocol['checkpoint']==old_manifest['checkpoint']
    model_config=Path(old_manifest['checkpoint']).with_name('config.json')
    assert chain.read(model_config)['model_family']=='nvs'
    runs={6:B6,8:B8}
    paths={6:B6.parent/'trajectory.json',8:B8/'trajectory.json'}
    trajectories={b:chain.read(p) for b,p in paths.items()}
    results={b:chain.read(root/'result.json') for b,root in runs.items()}
    for b in runs:
        assert results[b]['cameras_sha256']==chain.digest(paths[b])
        assert len(results[b]['frames'])==8
        assert chain.read(runs[b]/'visual_review.json')['state']=='accepted'
    sources=[]
    for j,(batch,local) in enumerate(SELECTION):
        r=results[batch]['frames'][local];t=trajectories[batch];image=Path(r['path']).resolve()
        assert chain.digest(image)==r['sha256']
        assert Image.open(image).size==(832,480)
        assert np.isclose(r['angle'],t['target_angles_deg'][local])
        sources.append(dict(index=j,role='generated',source_batch=batch,source_local_index=local,
            source_output_index=r['index'],original=str(image),sha256=r['sha256'],angle=r['angle'],
            w2c=t['target_w2c'][local],intrinsics=t['target_intrinsics'][local],
            source_run=str(runs[batch]),source_trajectory=str(paths[batch]),
            source_conditioning='Original batch6 matched-real-camera generation' if batch==6 else 'Batch8 generated using six real reference images; this source image itself is generated'))
    assert all(sources[i]['angle']<sources[i+1]['angle'] for i in range(5))
    assert [s['source_batch'] for s in sources]==[6]*3+[8]*3
    angles=np.arange(48,56)*5.625
    np.testing.assert_array_equal(old_target['target_angles_deg'],angles)
    target=dict(source_w2c=[r['w2c'] for r in sources],source_intrinsics=[r['intrinsics'] for r in sources],
        source_angles_deg=[r['angle'] for r in sources],target_w2c=old_target['target_w2c'],
        target_intrinsics=old_target['target_intrinsics'],target_angles_deg=old_target['target_angles_deg'],
        orbit_indices=list(range(48,56)),intrinsics_image_size=[832,480],camera_convention='OpenCV W2C',
        note='Two-sided batch7 interpolation from B6 local5/6/7 and pure-real-conditioned B8 local0/1/2. All six images are generated. Original batch7 target cameras unchanged.')
    geometry=chain.read(previous.GEOMETRY);center=np.array(geometry['center']);up=np.array(geometry['up']);up/=np.linalg.norm(up)
    anchor=np.array(trajectories[6]['target_w2c'][7]);ak=np.array(trajectories[6]['target_intrinsics'][7]);anchor_angle=trajectories[6]['target_angles_deg'][7]
    expected=previous.continuous_orbit(anchor,ak,anchor_angle,center,up,angles)
    np.testing.assert_allclose(target['target_w2c'],expected['target_w2c'],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(target['target_intrinsics'],expected['target_intrinsics'])
    next_camera=previous.continuous_orbit(anchor,ak,anchor_angle,center,up,[315])
    np.testing.assert_allclose(trajectories[8]['target_w2c'][0],next_camera['target_w2c'][0],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(trajectories[8]['target_intrinsics'][0],ak)
    cams=np.linalg.inv(np.array(target['target_w2c']));delta=cams[:,:3,3]-center
    heights=delta@up;radii=np.linalg.norm(delta-heights[:,None]*up,axis=1);distances=np.linalg.norm(delta,axis=1)
    np.testing.assert_allclose(distances,distances[0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(radii,radii[0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(heights,heights[0],atol=1e-12,rtol=0)
    audit=dict(state='passed',source_selection=[dict(batch=b,local_index=i) for b,i in SELECTION],
        source_angles_deg=target['source_angles_deg'],target_angles_deg=target['target_angles_deg'],
        target_cameras_identical_to_original_batch7=True,target_orbit_continues_to_batch8=True,
        horizontal_radius=float(radii[0]),height=float(heights[0]),distance=float(distances[0]),
        intrinsics=ak.tolist(),batch6_to_batch7_angle_step_deg=float(angles[0]-anchor_angle),
        batch7_to_batch8_angle_step_deg=5.625,source_gap_across_missing_batch_deg=float(sources[3]['angle']-sources[2]['angle']),
        limitations='This is a two-sided generated-reference interpolation test. Existing source appearance may conflict; requested-camera continuity does not guarantee generated-image geometry or continuity.')
    weights=old_manifest['weights']
    for r in weights:
        st=Path(r['path']).stat();assert [st.st_size,st.st_mtime_ns]==[r['size'],r['mtime_ns']]
    files=[Path(__file__).resolve(),Path(previous.__file__).resolve(),Path(chain.__file__).resolve(),
        previous.GEOMETRY,model_config,B6.parent/'protocol.json',OLD7/'manifest.json',OLD7/'trajectory.json',
        B8/'manifest.json',B6/'result.json',B8/'result.json',paths[6],paths[8]]
    files += [Path(r['original']) for r in sources]
    # Save immutable design before producing the generic inference manifest.
    for name,value in [('trajectory.json',target),('camera_design/audit.json',audit)]:
        previous.write_frozen_json(output/name,value)
        files.append(output/name)
    manifest=dict(batch=7,indices=list(range(48,56)),angles=angles.tolist(),sources=sources,
        source_selection=[list(p) for p in SELECTION],
        reference_policy='Three batch6 images05/06/07 plus three generated pure-real-conditioned batch8 images56/57/58; no previous batch7 images, no direct real references, no3DGS renders.',
        checkpoint=old_manifest['checkpoint'],omega=old_manifest['omega'],base=old_manifest['base'],
        seed=args.seed,steps=args.steps,resolution=[832,480],vram_limit_gib=args.vram_limit_gib,weights=weights,
        seed_matches_original_batch7=args.seed==old_manifest['seed'],steps_match_original_batch7=args.steps==old_manifest['steps'],
        target_cameras_exactly_match_original_batch7=True,baseline=str(OLD7),
        frozen_files=[dict(path=str(p),sha256=chain.digest(p)) for p in dict.fromkeys(files)])
    previous.write_frozen_json(output/'manifest.json',manifest)
    for r in sources:chain.frozen_copy(Path(r['original']),output/'source_views'/f"{r['index']:02d}.png")
    chain.validate(output)
    chain.montage([output/'source_views'/f'{i:02d}.png' for i in range(6)],
        [f"Ref{i} | B{r['source_batch']} local{r['source_local_index']} | {r['angle']:.3f} deg" for i,r in enumerate(sources)],
        output/'references6.jpg',3,(832,480))
    for p in [Path(__file__).resolve(),Path(previous.__file__).resolve(),Path(chain.__file__).resolve()]:
        chain.frozen_copy(p,output/'implementation'/p.name)
    if not (output/'status.json').exists():
        chain.save(output/'status.json',dict(state='prepared',batch=7,completed_frames=0))
    print('PREPARED_B6_B8_TO_B7',output,flush=True)
    print('REFERENCES',[(r['source_batch'],r['source_local_index'],r['angle']) for r in sources],flush=True)
    print('TARGETS',angles.tolist(),'seed',args.seed,'steps',args.steps,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--seed',type=int,default=20261029)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--vram-limit-gib',type=float,default=20)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    output=prepare(args)
    if not args.prepare_only:chain.infer(output)
