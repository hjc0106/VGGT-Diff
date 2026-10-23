"""Short overlapping-target diagnostic; generated targets are not hard RGB anchors."""
import argparse
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch

import vggtdiff_controlled_context as c
from vggtdiff_noise_control import frame_key,noise_digest
from vggtdiff_white_car import save,digest

CONTROL=c.OUT
ORBIT=CONTROL.with_name('vggtdiff_realvideo_orbit360_64_v2_batch8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['eight','five'],required=True);args=parser.parse_args()
    out=CONTROL.with_name('vggtdiff_realvideo_shared_context9_'+args.mode+'_v1');out.mkdir(exist_ok=True)
    c.e.validate_manifest(c.read(CONTROL/'manifest.json'))
    assert (CONTROL/'repro5/result.json').exists()
    for entry in c.read(CONTROL/'manifest.json')['weights']:
        st=Path(entry['path']).stat();assert st.st_size==entry['size'] and st.st_mtime_ns==entry['mtime_ns']
    if args.mode=='eight':
        assert (CONTROL/'context8/result.json').exists()
        cases={'sharedB8':[45,56.25,67.5,78.75,90,123.75,112.5,135]}
        reference=CONTROL/'context8'
    else:
        cases={'sharedA5':[45,56.25,78.75,90,135],'sharedB5':[45,90,101.25,123.75,135]}
        reference=CONTROL/'repro5'
    baseline=c.read(reference/'result.json');bank=torch.load(CONTROL/'noise_bank.pt',map_location='cpu',weights_only=True)
    keys=c.read(CONTROL/'noise_bank.json')['keys'];extra={}
    for a in sorted({a for values in cases.values() for a in values}):
        key=frame_key(a)
        if key in keys:continue
        seed=20261020+int(round(a*1000));gen=torch.Generator('cuda').manual_seed(seed)
        noise=torch.randn((1,16,1,60,104),generator=gen,device='cuda',dtype=torch.float32).to(bank.dtype).cpu()
        bank=torch.cat([bank,noise],dim=2);keys.append(key);extra[key]=seed
    p=out/'noise_bank.pt'
    if p.exists():assert torch.equal(torch.load(p,map_location='cpu',weights_only=True),bank)
    else:torch.save(bank,p)
    save(out/'noise_bank.json',dict(keys=keys,path=str(p),sha256=digest(p),extra_camera_seeds=extra,
        frame_hashes={key:noise_digest(bank[:,:,i:i+1]) for i,key in enumerate(keys)}))
    frozen=[Path(__file__).resolve(),CONTROL/'manifest.json',CONTROL/'noise_bank.pt',reference/'result.json']+[Path(f['path']) for f in baseline['frames']]
    manifest=dict(mode=args.mode,control=str(CONTROL),reference_case=str(reference),cases=cases,seed=20261020,steps=50,
        resolution=[832,480],vram_limit_gib=20,semantics='Shared target cameras and exact noise, not generated RGB conditioning or hard-clamped target latents',
        frozen_files=[dict(path=str(p),sha256=digest(p)) for p in frozen])
    if (out/'manifest.json').exists():assert c.read(out/'manifest.json')==manifest
    else:save(out/'manifest.json',manifest)
    c.e.validate_manifest(manifest);shutil.copyfile(__file__,out/'runner_snapshot.py')
    c.OUT=out;c.CASES=dict(c.CASES,**cases)
    pending=[name for name in cases if not (out/name/'result.json').exists()]
    if pending:
        from vggtdiff import VGGTDiff
        parent=c.read(c.PARENT/'inference_manifest.json')
        save(out/'status.json',dict(state='loading',cases=pending))
        model=VGGTDiff(parent['checkpoint'],parent['omega'],base_model_dir=parent['base'],device='cuda',vram_limit_gib=20,training_mode='nvs',load_full_checkpoint_directly=True)
        for name in pending:c.run_case(model,name,bank,keys)
    results=[baseline]+[c.read(out/name/'result.json') for name in cases]
    for result in results:
        for frame in result['frames']:assert digest(Path(frame['path']))==frame['sha256']
        noise_file=reference/'initial_noise.pt' if result is baseline else out/result['case']/'initial_noise.pt'
        assert digest(noise_file)==result['noise_file_sha256']
        for key,value in result['noise']['frame_hashes'].items():
            assert value==c.read(out/'noise_bank.json')['frame_hashes'][key]
    shared=[]
    for left_i,left in enumerate(results):
        for right in results[left_i+1:]:
            lm={f['angle']:f for f in left['frames']};rm={f['angle']:f for f in right['frames']}
            for angle in sorted(lm.keys()&rm.keys()):
                key=frame_key(angle);assert left['noise']['frame_hashes'][key]==right['noise']['frame_hashes'][key]
                ia=np.asarray(Image.open(lm[angle]['path']).convert('RGB'),dtype=float)
                ib=np.asarray(Image.open(rm[angle]['path']).convert('RGB'),dtype=float)
                shared.append(dict(left_case=left['case'],right_case=right['case'],angle=angle,
                    rgb_mae_01=float(np.abs(ia-ib).mean()/255),noise_identical=True))
    angles=np.linspace(45,135,9).tolist();records=[];(out/'raw').mkdir(exist_ok=True)
    for i,angle in enumerate(angles):
        if args.mode=='eight':
            chosen=baseline if angle<=101.25 else results[1]
        elif angle in c.CASES['repro5']:chosen=baseline
        else:chosen=results[1] if angle<90 else results[2]
        f=next(f for f in chosen['frames'] if f['angle']==angle);src=Path(f['path']);assert digest(src)==f['sha256']
        dst=out/'raw'/f'{i:02d}.png';shutil.copyfile(src,dst)
        records.append(dict(index=i,angle=angle,path=str(dst),sha256=digest(dst),origin_case=chosen['case'],origin_path=str(src)))
    save(out/'trajectory.json',c.trajectory(angles))
    c.e.contact([Path(f['path']) for f in records],[f'{f["index"]:02d}: {f["angle"]:g} deg | {f["origin_case"]}' for f in records],out/'short9_contact.jpg',columns=3)
    with imageio.get_writer(str(out/'short9.mp4'),fps=3,macro_block_size=16) as video, imageio.get_writer(str(out/'original64_vs_short9.mp4'),fps=3,macro_block_size=16) as comparison:
        for f in records:
            angle=f['angle'];index=int(round(angle/5.625));im=Image.open(f['path']).convert('RGB')
            panel=Image.new('RGB',(832,512),'#181818');panel.paste(im,(0,32));ImageDraw.Draw(panel).text((10,10),f'{angle:g} deg | {f["origin_case"]} | exact camera-keyed noise',fill='white');video.append_data(np.array(panel))
            old=Image.open(ORBIT/'raw'/f'{index:02d}.png').convert('RGB');pair=Image.new('RGB',(1664,512),'#181818');pair.paste(old,(0,32));pair.paste(im,(832,32));draw=ImageDraw.Draw(pair);draw.text((10,10),f'Previous full orbit: {index:02d} | {angle:g} deg',fill='white');draw.text((842,10),f'Shared-target context: {angle:g} deg',fill='white');comparison.append_data(np.array(pair))
    overlap_paths=[];overlap_labels=[]
    for f in baseline['frames']:
        if any(f['angle'] in cases[n] for n in cases):
            overlap_paths.append(Path(f['path']));overlap_labels.append(f'{baseline["case"]}: {f["angle"]:g}')
            for result in results[1:]:
                other=next((x for x in result['frames'] if x['angle']==f['angle']),None)
                if other:overlap_paths.append(Path(other['path']));overlap_labels.append(f'{result["case"]}: {f["angle"]:g}')
    c.e.contact(overlap_paths,overlap_labels,out/'shared_camera_comparison.jpg',columns=len(results))
    c.e.validate_manifest(manifest)
    for entry in c.read(CONTROL/'manifest.json')['weights']:
        st=Path(entry['path']).stat();assert st.st_size==entry['size'] and st.st_mtime_ns==entry['mtime_ns']
    save(out/'result.json',dict(frames=records,shared_camera_comparisons=shared,visual_review='pending',
        caveat='Shared target/noise overlap is a hypothesis to test, not a guarantee of consistency. Previous-full-orbit comparison also changes initial noise and target composition.'))
    save(out/'status.json',dict(state='completed_pending_visual_review',frames=9,shared_camera_pairs=len(shared)))
    print('COMPLETED_SHARED_CONTEXT_SHORT',json.dumps(shared),flush=True)


if __name__=='__main__':main()
