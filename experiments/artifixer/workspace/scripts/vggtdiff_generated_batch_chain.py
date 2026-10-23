"""Generate one orbit batch from six selected images of a prior generated run.

Uses the prior output cameras verbatim, not nominal uniform batch-six angles.
Preparation is CPU-only. Without --prepare-only, run eight-target NVS inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'thirdparty/VGGT-Diff'),str(REPO),str(REPO/'worldact')]
from vggtdiff_white_car import BASE, digest, save
from vggtdiff_latest_coverage_ab import frozen_copy, montage


def read(path):
    return json.loads(Path(path).read_text())


def validate(output):
    manifest=read(output/'manifest.json')
    for row in manifest['frozen_files']:
        if digest(Path(row['path']))!=row['sha256']:
            raise ValueError(f"Frozen input changed: {row['path']}")
    for row in manifest['sources']:
        if digest(output/'source_views'/f"{row['index']:02d}.png")!=row['sha256']:
            raise ValueError('Copied reference image changed')
    for row in manifest['weights']:
        stat=Path(row['path']).stat()
        if [stat.st_size,stat.st_mtime_ns]!=[row['size'],row['mtime_ns']]:
            raise ValueError(f"Weight changed: {row['path']}")
    return manifest,read(output/'trajectory.json')


def prepare(args):
    source=args.source_run.resolve()
    source_trajectory=(args.source_trajectory or source.parent/'trajectory.json').resolve()
    protocol_path=source.parent/'protocol.json'
    protocol=read(protocol_path)
    original=read(source_trajectory)
    records=read(source/'result.json')['frames']
    by_index={r['index']:r for r in records}
    selected=args.reference_indices
    if len(set(selected))!=6 or any(i not in by_index for i in selected):
        raise ValueError('Choose six distinct local output indices from the source run')
    if (source/'visual_review.json').exists() and read(source/'visual_review.json')['state']!='accepted':
        raise ValueError('Source run has not passed its visual review')
    orbit=read(args.orbit_trajectory)
    indices=list(range((args.batch-1)*8,args.batch*8))
    angles=[orbit['target_angles_deg'][i] for i in indices]
    if not np.allclose(angles,np.array(indices)*5.625):
        raise ValueError('Orbit angle/index mismatch')
    config=Path(protocol['checkpoint']).with_name('config.json')
    if read(config)['model_family']!='nvs': raise ValueError('An NVS checkpoint is required')
    sources=[]
    for j,i in enumerate(selected):
        record=by_index[i]
        path=Path(record['path']).resolve()
        if digest(path)!=record['sha256']: raise ValueError(f'Source image changed: {path}')
        if Image.open(path).size!=(832,480): raise ValueError(f'Invalid source dimensions: {path}')
        if not np.isclose(record['angle'],original['target_angles_deg'][i]):
            raise ValueError('Source image/camera angle mismatch')
        sources.append(dict(index=j,source_output_index=i,role='generated',original=str(path),
            angle=record['angle'],sha256=record['sha256'],w2c=original['target_w2c'][i],
            intrinsics=original['target_intrinsics'][i]))
    trajectory=dict(source_w2c=[r['w2c'] for r in sources],
        source_intrinsics=[r['intrinsics'] for r in sources],
        target_w2c=[orbit['target_w2c'][i] for i in indices],
        target_intrinsics=[orbit['target_intrinsics'][i] for i in indices],
        source_angles_deg=[r['angle'] for r in sources],target_angles_deg=angles,orbit_indices=indices,
        intrinsics_image_size=[832,480],camera_convention='OpenCV W2C',
        note='Generated references only; preserve actual matched-GT source poses/K; targets use the existing 64-view orbit.')
    for w,k in zip(trajectory['source_w2c']+trajectory['target_w2c'],trajectory['source_intrinsics']+trajectory['target_intrinsics']):
        w,k=np.asarray(w),np.asarray(k)
        if w.shape!=(4,4) or k.shape!=(3,3) or not np.isfinite(w).all() or not np.isfinite(k).all():
            raise ValueError('Invalid camera shape or nonfinite values')
        if not np.allclose(w[:3,:3]@w[:3,:3].T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(w[:3,:3]),1,atol=1e-5):
            raise ValueError('Invalid camera rotation')
        if k[0,0]<=0 or k[1,1]<=0: raise ValueError('Nonpositive focal length')
    if not 0<args.steps or args.vram_limit_gib<0: raise ValueError('Invalid steps or VRAM limit')
    output=(args.output or BASE/f'vggtdiff_batch06gen_to_batch{args.batch:02d}_v1').resolve()
    protected=[source,source.parent,args.orbit_trajectory.resolve().parent,BASE.resolve()]
    if output in protected or source.is_relative_to(output): raise ValueError('Output overlaps source data')
    files=[source/'result.json',source_trajectory,protocol_path,args.orbit_trajectory.resolve(),config,Path(__file__).resolve()]
    files += [Path(r['original']) for r in sources]
    weights=[]
    for p in [Path(protocol['checkpoint']),Path(protocol['omega'])]:
        st=p.stat();weights.append(dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns))
    manifest=dict(batch=args.batch,indices=indices,angles=angles,source_run=str(source),
        source_trajectory=str(source_trajectory),source_indices=selected,sources=sources,
        reference_policy='Exactly six generated images; no real image conditioning and no 3DGS renders.',
        checkpoint=protocol['checkpoint'],omega=protocol['omega'],base=protocol['base'],
        seed=args.seed,steps=args.steps,resolution=[832,480],vram_limit_gib=args.vram_limit_gib,
        weights=weights,frozen_files=[dict(path=str(p),sha256=digest(p)) for p in files])
    for name,value in [('manifest.json',manifest),('trajectory.json',trajectory)]:
        if (output/name).exists() and read(output/name)!=value:
            raise ValueError(f'Existing experiment differs: {output/name}; choose a new --output')
    output.mkdir(parents=True,exist_ok=True)
    for r in sources:
        frozen_copy(Path(r['original']),output/'source_views'/f"{r['index']:02d}.png")
    save(output/'trajectory.json',trajectory);save(output/'manifest.json',manifest)
    validate(output)
    if not (output/'result.json').exists(): save(output/'status.json',dict(state='prepared',batch=args.batch))
    print(json.dumps(dict(output=str(output),source_indices=selected,
        source_angles_deg=trajectory['source_angles_deg'],target_indices=indices,target_angles_deg=angles,
        seed=args.seed,steps=args.steps),indent=2),flush=True)
    return output


def infer(output):
    manifest,t=validate(output)
    if (output/'result.json').exists(): raise FileExistsError(f'Completed output exists: {output}')
    import torch
    import imageio.v2 as imageio
    from vggtdiff import VGGTDiff
    save(output/'status.json',dict(state='loading',batch=manifest['batch']))
    model=VGGTDiff(manifest['checkpoint'],manifest['omega'],base_model_dir=manifest['base'],
        device='cuda',vram_limit_gib=manifest['vram_limit_gib'],training_mode='nvs',load_full_checkpoint_directly=True)
    if model.pipe.dit.training: raise RuntimeError('DiT must be in eval mode')
    images=[Image.open(output/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
    w=np.asarray(t['source_w2c']+t['target_w2c'],np.float32)
    k=np.asarray(t['source_intrinsics']+t['target_intrinsics'],np.float32)
    assert w.shape==(14,4,4) and k.shape==(14,3,3)
    save(output/'status.json',dict(state='generating',batch=manifest['batch']))
    torch.cuda.reset_peak_memory_stats();start=time.time()
    predictions=model.generate(images,w,k,height=480,width=832,steps=manifest['steps'],seed=manifest['seed'])
    if len(predictions)!=8: raise ValueError('Expected eight predictions')
    (output/'raw').mkdir(exist_ok=True);records=[]
    with imageio.get_writer(str(output/'prediction_8.mp4'),fps=2,macro_block_size=16) as writer:
        for image,index,angle in zip(predictions,manifest['indices'],manifest['angles']):
            if image.size!=(832,480): raise ValueError('Invalid prediction dimensions')
            path=output/'raw'/f'{index:02d}.png';image.save(path);writer.append_data(np.asarray(image))
            records.append(dict(index=index,angle=angle,path=str(path),sha256=digest(path)))
    montage([Path(r['path']) for r in records],[f"{r['index']:02d} | {r['angle']:.3f} deg" for r in records],output/'contact8.jpg',columns=4)
    validate(output)
    save(output/'result.json',dict(frames=records,seed=manifest['seed'],steps=manifest['steps'],
        elapsed_seconds=time.time()-start,max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        cameras_sha256=digest(output/'trajectory.json'),dit_eval=not model.pipe.dit.training))
    save(output/'status.json',dict(state='completed_pending_visual_review',batch=manifest['batch'],completed_frames=8))
    print('COMPLETED',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,default=BASE/'crossview_geometry_matched_gt_v1/seed_20261028')
    p.add_argument('--source-trajectory',type=Path)
    p.add_argument('--reference-indices',type=int,nargs=6,default=[0,1,3,4,6,7])
    p.add_argument('--orbit-trajectory',type=Path,default=BASE/'vggtdiff_realvideo_orbit360_64_latest_refs_v1/trajectory.json')
    p.add_argument('--batch',type=int,choices=range(1,9),default=7)
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--vram-limit-gib',type=float,default=20)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();output=prepare(args)
    if not args.prepare_only: infer(output)

if __name__=='__main__':main()
