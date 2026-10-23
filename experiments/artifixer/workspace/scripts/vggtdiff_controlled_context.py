"""Reproduce the five-view experiment and isolate target-set/noise effects."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
import vggtdiff_realvideo_unseen as e
from vggtdiff_noise_control import frame_key,noise_digest,select_noise

PARENT=e.OUT.with_name('vggtdiff_realvideo_unseen_right_near_v2')
OUT=PARENT.with_name('vggtdiff_realvideo_controlled_context_v1')
CASES={'repro5':[45,67.5,90,112.5,135], 'count4':[45,67.5,90,112.5],
       'context8':[45,56.25,67.5,78.75,90,101.25,112.5,135]}
SOURCE_KEYS=[f'source:{i:02d}' for i in range(6)]


def read(p):return json.loads(p.read_text())


def prepare():
    original=read(PARENT/'inference_manifest.json');e.validate_manifest(original)
    OUT.mkdir(exist_ok=True)
    paths=[PARENT/'inference_manifest.json',PARENT/'trajectory.json',PARENT/'geometry_check.json']
    paths += [PARENT/'source_views'/f'{i:02d}.png' for i in range(6)]
    paths += [Path(__file__).resolve(),e.REPO/'scripts/vggtdiff_noise_control.py']
    paths += [e.REPO/'thirdparty/VGGT-Diff'/n for n in ['vggtdiff/model.py','vggtdiff/omega.py','vggtdiff/camera.py','diffsynth/pipelines/wan_video.py','diffsynth/models/wan_video_dit.py']]
    manifest=dict(parent=str(PARENT),cases=CASES,steps=50,seed=20261020,resolution=[832,480],vram_limit_gib=20,
        source_noise='Original five-view tensor, keyed by source index',target_noise='Public cameras reuse exact original tensor slices; extra cameras get separately recorded noise',
        frozen_files=[dict(path=str(p),sha256=e.digest(p)) for p in paths],weights=original['weights'],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(),
        versions={k:importlib.metadata.version(k) for k in ['torch','numpy','Pillow','safetensors','transformers']})
    p=OUT/'manifest.json'
    if p.exists():assert read(p)==manifest,'Frozen experiment changed'
    else:e.save(p,manifest)
    e.validate_manifest(manifest)
    for w in manifest['weights']:
        st=Path(w['path']).stat();assert st.st_size==w['size'] and st.st_mtime_ns==w['mtime_ns']
    shutil.copyfile(__file__,OUT/'runner_snapshot.py')
    shutil.copyfile(e.REPO/'scripts/vggtdiff_noise_control.py',OUT/'noise_control_snapshot.py')
    return original,manifest


def trajectory(angles):
    source=read(PARENT/'trajectory.json');g=read(PARENT/'geometry_check.json')
    center=np.array(g['center']);front=np.array(g['front']);left=np.array(g['left']);up=np.array(g['up'])
    ws=[]
    for a in angles:
        theta=np.deg2rad(a)
        p=center+g['target_radius']*(np.cos(theta)*front-np.sin(theta)*left)+g['target_height_relative_to_body_center']*up
        ws.append(e.look_at(p,center,up).tolist())
    for i,a in enumerate(angles):
        if a in CASES['repro5']:
            assert np.allclose(ws[i],source['target_w2c'][CASES['repro5'].index(a)],rtol=0,atol=1e-12)
    return dict(source_w2c=source['source_w2c'],source_intrinsics=source['source_intrinsics'],target_w2c=ws,
        target_intrinsics=[source['target_intrinsics'][0]]*len(angles),target_angles_deg=angles)


def make_bank():
    noise=torch.load(OUT/'repro5/initial_noise.pt',map_location='cpu',weights_only=True)
    keys=SOURCE_KEYS+[frame_key(a) for a in CASES['repro5']]
    assert noise.shape==(1,16,11,60,104)
    extra=sorted(set(CASES['context8'])-set(CASES['repro5']))
    seeds={}
    for a in extra:
        seed=20261020+int(round(a*1000));seeds[frame_key(a)]=seed
        gen=torch.Generator('cuda').manual_seed(seed)
        frame=torch.randn((1,16,1,60,104),generator=gen,device='cuda',dtype=torch.float32).to(noise.dtype).cpu()
        noise=torch.cat([noise,frame],dim=2);keys.append(frame_key(a))
    p=OUT/'noise_bank.pt'
    if p.exists():assert torch.equal(noise,torch.load(p,map_location='cpu',weights_only=True))
    else:torch.save(noise,p)
    e.save(OUT/'noise_bank.json',dict(path=str(p),sha256=e.digest(p),keys=keys,extra_camera_seeds=seeds,
        frame_hashes={key:noise_digest(noise[:,:,i:i+1]) for i,key in enumerate(keys)}))
    return noise,keys


def compare_old(images,angles):
    result=[]
    for im,a in zip(images,angles):
        if a not in CASES['repro5']:continue
        old=np.asarray(Image.open(PARENT/'pilot/raw'/f'{CASES["repro5"].index(a):02d}.png').convert('RGB'),dtype=float)
        new=np.asarray(im,dtype=float)
        result.append(dict(angle=a,rgb_mae_01=float(np.abs(old-new).mean()/255),pixel_identical=bool(np.array_equal(old,new))))
    return result


def run_case(model,name,bank=None,keys=None):
    dest=OUT/name;dest.mkdir(exist_ok=True);angles=CASES[name];t=trajectory(angles)
    e.save(dest/'trajectory.json',t);wanted=SOURCE_KEYS+[frame_key(a) for a in angles]
    original_noise=model.pipe.generate_noise;calls=[]
    def noise_hook(shape,**kwargs):
        if calls:raise RuntimeError('Unexpected second diffusion-noise initialization')
        if name=='repro5':noise=original_noise(shape,**kwargs)
        else:
            noise=select_noise(bank,keys,wanted).to(device=kwargs.get('device') or model.pipe.device,dtype=kwargs.get('torch_dtype') or model.pipe.torch_dtype)
        assert tuple(noise.shape)==tuple(shape)==(1,16,len(wanted),60,104)
        torch.save(noise.cpu(),dest/'initial_noise.pt')
        calls.append(dict(keys=wanted,shape=list(shape),tensor_sha256=noise_digest(noise),
            frame_hashes={key:noise_digest(noise[:,:,i:i+1]) for i,key in enumerate(wanted)}))
        return noise
    model.pipe.generate_noise=noise_hook
    images=[Image.open(PARENT/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
    w=np.array(t['source_w2c']+t['target_w2c'],dtype=np.float32);k=np.array(t['source_intrinsics']+t['target_intrinsics'],dtype=np.float32)
    e.save(OUT/'status.json',dict(state='generating',case=name,angles=angles,steps=50))
    print('START_CASE',name,angles,flush=True);start=time.time();torch.cuda.reset_peak_memory_stats()
    try:outputs=model.generate(images,w,k,height=480,width=832,steps=50,seed=20261020)
    finally:model.pipe.generate_noise=original_noise
    assert len(outputs)==len(angles) and len(calls)==1
    (dest/'raw').mkdir(exist_ok=True);records=[]
    for i,(a,im) in enumerate(zip(angles,outputs)):
        assert im.size==(832,480)
        path=dest/'raw'/f'{i:02d}.png';im.save(path)
        records.append(dict(index=i,angle=a,path=str(path),sha256=e.digest(path)))
    e.contact([Path(x['path']) for x in records],[f'{name}: {a:g} deg' for a in angles],dest/'contact.jpg',columns=3)
    if name!='repro5':
        expected=read(OUT/'noise_bank.json')['frame_hashes']
        assert all(value==expected[key] for key,value in calls[0]['frame_hashes'].items())
    result=dict(case=name,frames=records,noise=calls[0],noise_file_sha256=e.digest(dest/'initial_noise.pt'),
        elapsed_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        old_five_comparison=compare_old(outputs,angles),visual_review='pending')
    e.save(dest/'result.json',result);print('COMPLETED_CASE',name,json.dumps(result['old_five_comparison']),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES));args=parser.parse_args()
    original,manifest=prepare()
    pending=[]
    for name in args.cases:
        p=OUT/name/'result.json'
        if p.exists():
            result=read(p)
            for frame in result['frames']:assert e.digest(Path(frame['path']))==frame['sha256']
            assert e.digest(OUT/name/'initial_noise.pt')==result['noise_file_sha256']
        else:pending.append(name)
    if pending:
        from vggtdiff import VGGTDiff
        e.save(OUT/'status.json',dict(state='loading',pending=pending))
        model=VGGTDiff(original['checkpoint'],original['omega'],base_model_dir=original['base'],device='cuda',vram_limit_gib=20,training_mode='nvs',load_full_checkpoint_directly=True)
        assert not model.pipe.dit.training
        for name in pending:
            if name=='repro5':run_case(model,name)
            else:
                if not (OUT/'repro5/result.json').exists():raise RuntimeError('Five-view reproduction must precede comparisons')
                repro=read(OUT/'repro5/result.json')
                if max(x['rgb_mae_01'] for x in repro['old_five_comparison'])>.025:
                    e.save(OUT/'status.json',dict(state='reproduction_mismatch',reason='Five-view reproduction differs from original; investigate before target-set ablations'))
                    print('STOP_REPRODUCTION_MISMATCH',flush=True);return
                bank,keys=make_bank();run_case(model,name,bank,keys)
    e.validate_manifest(manifest)
    e.save(OUT/'status.json',dict(state='controlled_cases_completed_pending_review',completed=[n for n in CASES if (OUT/n/'result.json').exists()]))
    print('COMPLETED_CONTROLLED_CASES',flush=True)


if __name__=='__main__':main()
