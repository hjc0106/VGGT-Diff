"""Batch8 NVS with one batch6 + five batch7 generated references.

Reference selection is the already visualized references.json, with each image
paired to its own generating camera. Targets rigidly continue batch7's last
camera at315..354.375deg. --prepare-only does no diffusion/model loading.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import vggtdiff_batch07_continuous as previous

chain=previous.chain
B6=chain.BASE/'crossview_geometry_matched_gt_v1/seed_20261028'
B7=previous.DEFAULT_OUTPUT
REFERENCE_PLAN=chain.BASE/'vggtdiff_batch08_mixed_refs_preview_v1/references.json'
OUT=chain.BASE/'vggtdiff_batch06_07gen_to_batch08_continuous_v1'
SELECTION=[(6,7),(7,1),(7,3),(7,4),(7,6),(7,7)]


def prepare(args):
    output=args.output.resolve()
    protected=[B6.resolve(),B7.resolve(),REFERENCE_PLAN.parent.resolve()]
    if output==chain.BASE.resolve() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be separate from source experiments')
    if args.steps<=0 or args.vram_limit_gib<0: raise ValueError('Invalid steps or VRAM limit')
    chain.validate(B7)
    for row in chain.read(B7/'camera_design/manifest.json')['files']:
        if chain.digest(Path(row['path']))!=row['sha256']: raise ValueError(f"Camera design changed: {row['path']}")
    runs={6:B6,7:B7};paths={6:B6.parent/'trajectory.json',7:B7/'trajectory.json'}
    trajectories={b:chain.read(p) for b,p in paths.items()}
    results={b:chain.read(p/'result.json') for b,p in runs.items()}
    for b in [6,7]:
        if results[b]['cameras_sha256']!=chain.digest(paths[b]): raise ValueError('Source camera hash mismatch')
        if chain.read(runs[b]/'visual_review.json')['state']!='accepted': raise ValueError('Source visual review missing')
        if len(results[b]['frames'])!=8: raise ValueError('Expected eight frames per source batch')
    plan=chain.read(REFERENCE_PLAN)['sources']
    if [(r['batch'],r['local_index']) for r in plan]!=SELECTION:
        raise ValueError('Reference plan differs from the six images already discussed and visualized')
    sources=[]
    for j,row in enumerate(plan):
        b,i=row['batch'],row['local_index'];record=results[b]['frames'][i];t=trajectories[b]
        path=Path(record['path']).resolve()
        if str(path)!=row['image'] or chain.digest(path)!=row['sha256'] or row['sha256']!=record['sha256']:
            raise ValueError(f'Reference plan/image mismatch: {path}')
        with Image.open(path) as image:
            if image.size!=(832,480): raise ValueError('Invalid image dimensions')
        if not np.isclose(row['angle_deg'],t['target_angles_deg'][i]) or not np.isclose(record['angle'],t['target_angles_deg'][i]):
            raise ValueError('Reference angle/image/camera mismatch')
        sources.append(dict(index=j,role='generated',source_batch=b,source_local_index=i,
            source_output_index=record['index'],original=str(path),sha256=record['sha256'],
            angle=record['angle'],w2c=t['target_w2c'][i],intrinsics=t['target_intrinsics'][i],
            source_run=str(runs[b]),source_trajectory=str(paths[b])))
    anchor_w=np.asarray(trajectories[7]['target_w2c'][7]);anchor_k=np.asarray(trajectories[7]['target_intrinsics'][7])
    anchor_angle=trajectories[7]['target_angles_deg'][7]
    if not np.isclose(anchor_angle,309.375): raise ValueError('Unexpected batch7 boundary angle')
    geometry=chain.read(previous.GEOMETRY);center=np.asarray(geometry['center']);up=np.asarray(geometry['up']);up=up/np.linalg.norm(up)
    angles=np.arange(56,64)*5.625
    targets=previous.continuous_orbit(anchor_w,anchor_k,anchor_angle,center,up,angles)
    # Repeated continuation must agree with the original batch6-anchored circle.
    original_circle=chain.read(B7/'camera_design/target_orbit64.json')
    np.testing.assert_allclose(targets['target_w2c'],np.asarray(original_circle['target_w2c'])[56:64],atol=1e-12,rtol=0)
    assert np.array_equal(targets['target_intrinsics'],np.asarray(original_circle['target_intrinsics'])[56:64])
    offset=np.linalg.inv(anchor_w)[:3,3]-center
    height=float(offset@up);radius=float(np.linalg.norm(offset-height*up));distance=float(np.linalg.norm(offset))
    anchor_cc=anchor_w@np.r_[center,1];checks=[]
    for i,(w,k,angle) in enumerate(zip(targets['target_w2c'],targets['target_intrinsics'],angles)):
        w,k=np.asarray(w),np.asarray(k);pos=np.linalg.inv(w)[:3,3];delta=pos-center;h=float(delta@up);r=float(np.linalg.norm(delta-h*up))
        azimuth=float(np.degrees(np.arctan2(-delta@geometry['left'],delta@geometry['front']))%360)
        if abs(((azimuth-angle+180)%360)-180)>1e-4: raise ValueError('Wrong target azimuth')
        np.testing.assert_allclose([r,h,np.linalg.norm(delta)],[radius,height,distance],atol=1e-12,rtol=0)
        np.testing.assert_allclose(w@np.r_[center,1],anchor_cc,atol=1e-12,rtol=0)
        if not np.array_equal(k,anchor_k): raise ValueError('Target intrinsics changed')
        checks.append(dict(index=56+i,angle_deg=float(angle),measured_azimuth_deg=azimuth,
            horizontal_radius=r,height=h,distance=float(np.linalg.norm(delta)),center_depth=float((w@np.r_[center,1])[2]),
            central_scale_ratio_to_anchor=float(anchor_cc[2]/(w@np.r_[center,1])[2])))
    trajectory=dict(source_w2c=[r['w2c'] for r in sources],source_intrinsics=[r['intrinsics'] for r in sources],
        source_angles_deg=[r['angle'] for r in sources],target_w2c=targets['target_w2c'],target_intrinsics=targets['target_intrinsics'],
        target_angles_deg=angles.tolist(),orbit_indices=list(range(56,64)),intrinsics_image_size=[832,480],camera_convention='OpenCV W2C',
        note='Mixed generated references: B6 local7, B7 local1/3/4/6/7. Targets rigidly continue B7 local7 without distance, height, roll or K jumps.')
    audit=dict(state='passed',anchor_batch=7,anchor_local_index=7,anchor_angle_deg=anchor_angle,
        boundary_angle_change_deg=float(angles[0]-anchor_angle),horizontal_radius=radius,height=height,distance=distance,
        anchor_w2c=anchor_w.tolist(),intrinsics=anchor_k.tolist(),center=center.tolist(),up=up.tolist(),targets=checks,
        matches_original_batch6_anchored_orbit=True,units='Existing relative world units; not calibrated meters')
    protocol=chain.read(B6.parent/'protocol.json')
    model_config=Path(protocol['checkpoint']).with_name('config.json')
    if chain.read(model_config)['model_family']!='nvs': raise ValueError('NVS checkpoint required')
    weights=[]
    for p in [Path(protocol['checkpoint']),Path(protocol['omega'])]:
        st=p.stat();weights.append(dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns))
    # Freeze source artifacts, the geometry helpers and the exact output cameras.
    files=[REFERENCE_PLAN,previous.GEOMETRY,B6.parent/'protocol.json',model_config,
        B6/'result.json',B7/'result.json',paths[6],paths[7],B7/'camera_design/target_orbit64.json',
        Path(__file__).resolve(),Path(previous.__file__).resolve(),Path(chain.__file__).resolve()]
    files += [Path(r['original']) for r in sources]
    manifest=dict(batch=8,indices=list(range(56,64)),angles=angles.tolist(),sources=sources,
        reference_plan=str(REFERENCE_PLAN),reference_policy='One generated image from batch6 and five from batch7, exactly as previously visualized; no real conditioning images or GS renders.',
        checkpoint=protocol['checkpoint'],omega=protocol['omega'],base=protocol['base'],
        seed=args.seed,steps=args.steps,resolution=[832,480],vram_limit_gib=args.vram_limit_gib,weights=weights,
        frozen_files=[dict(path=str(p),sha256=chain.digest(p)) for p in dict.fromkeys(files)])
    # Hash camera JSON exactly as chain.save serializes it, then validate before mutation.
    artifacts=[('trajectory.json',trajectory),('camera_design/audit.json',audit)]
    for name,value in artifacts:
        payload=(json.dumps(value,indent=2)+'\n').encode()
        manifest['frozen_files'].append(dict(path=str(output/name),sha256=hashlib.sha256(payload).hexdigest()))
    for name,value in artifacts+[('manifest.json',manifest)]:
        if (output/name).exists() and chain.read(output/name)!=value:
            raise ValueError(f'Existing experiment differs: {output/name}; choose a new --output')
    for name,value in artifacts+[('manifest.json',manifest)]:
        previous.write_frozen_json(output/name,value)
    for r in sources: chain.frozen_copy(Path(r['original']),output/'source_views'/f"{r['index']:02d}.png")
    chain.validate(output)
    chain.montage([output/'source_views'/f'{i:02d}.png' for i in range(6)],
        [f"Ref{i} | B{r['source_batch']} local{r['source_local_index']} | {r['angle']:.3f} deg" for i,r in enumerate(sources)],
        output/'references6.jpg',columns=3)
    for p in [Path(__file__).resolve(),Path(previous.__file__).resolve(),Path(chain.__file__).resolve()]:
        chain.frozen_copy(p,output/'implementation'/p.name)
    if not (output/'result.json').exists(): chain.save(output/'status.json',dict(state='prepared',batch=8,completed_frames=0))
    print('BATCH8_PREPARED',output,flush=True)
    print('REFERENCES',[(r['source_batch'],r['source_local_index'],r['angle']) for r in sources],flush=True)
    print('TARGET_ANGLES',angles.tolist(),flush=True)
    print('CONTINUITY',dict(radius=radius,height=height,distance=distance,K=anchor_k.tolist(),boundary_step_deg=angles[0]-anchor_angle),flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--vram-limit-gib',type=float,default=20)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();output=prepare(args)
    if not args.prepare_only: chain.infer(output)

if __name__=='__main__':main()
