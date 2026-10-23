"""Generate a full 360-degree real-video-conditioned 64-camera orbit."""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import vggtdiff_realvideo_unseen as e

PARENT=e.OUT.with_name('vggtdiff_realvideo_unseen_right_near_v2')
OUT=PARENT.with_name('vggtdiff_realvideo_orbit360_64_v2_batch8')


def read(p):
    return json.loads(p.read_text())


def prepare():
    e.validate_manifest(read(PARENT/'inference_manifest.json'))
    OUT.mkdir(exist_ok=True)
    g=read(PARENT/'geometry_check.json');t=read(PARENT/'trajectory.json')
    axes=np.array([g['front'],g['left'],g['up']]).T
    center=np.array(g['center']);angles=np.linspace(0,360,64,endpoint=False)
    positions=[center+g['target_radius']*(np.cos(np.deg2rad(a))*axes[:,0]-np.sin(np.deg2rad(a))*axes[:,1])+g['target_height_relative_to_body_center']*axes[:,2] for a in angles]
    ws=[e.look_at(p,center,axes[:,2]) for p in positions]
    k=np.array(t['target_intrinsics'][0]);assert np.allclose(t['target_intrinsics'],k)
    lo,hi=np.array(g['observed_car_bounds_local'])
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@axes.T+g['ground_origin']
    bounds=[]
    for p,w in zip(positions,ws):
        uv,z=e.project(corners,w,k);assert z.min()>0
        b=[*uv.min(0),*uv.max(0)];assert b[0]>0 and b[1]>0 and b[2]<832 and b[3]<480
        bounds.append(b)
    trajectory=dict(source_w2c=t['source_w2c'],source_intrinsics=t['source_intrinsics'],target_w2c=np.array(ws).tolist(),target_intrinsics=[k.tolist()]*64,target_angles_deg=angles.tolist())
    path=OUT/'trajectory.json'
    if path.exists():assert read(path)==trajectory
    else:e.save(path,trajectory)
    (OUT/'source_views').mkdir(exist_ok=True)
    for i in range(6):
        src=PARENT/'source_views'/f'{i:02d}.png';dst=OUT/'source_views'/src.name
        if not dst.exists():shutil.copyfile(src,dst)
        assert e.digest(src)==e.digest(dst)
    e.save(OUT/'camera_checks.json',dict(parent=str(PARENT),count=64,full_360_orbit=True,azimuth_step_degrees=5.625,first_last_duplicate=False,angles_deg=angles.tolist(),observed_bounds_pixels=bounds,radius=g['target_radius'],height=g['target_height_relative_to_body_center'],source_cameras_unchanged=True,limitation='Observed partial bounds are not complete hidden-side geometry'))
    manifest=dict(parent=str(PARENT),target_count=64,batch_size=8,batch_count=8,seed_per_batch=20261020,steps=50,resolution=[832,480],vram_limit_gib=40,global_joint_64=False,frozen_files=[dict(path=str(p),sha256=e.digest(p)) for p in [OUT/'trajectory.json',Path(__file__).resolve()]+list(sorted((OUT/'source_views').glob('*.png')))],parent_manifest_sha256=e.digest(PARENT/'inference_manifest.json'))
    path=OUT/'inference_manifest.json'
    if path.exists():assert read(path)==manifest
    else:e.save(path,manifest)
    shutil.copyfile(__file__,OUT/'inference_script_snapshot.py')
    return trajectory,manifest


def assemble():
    import imageio.v2 as imageio
    t=read(OUT/'trajectory.json');records=[]
    overview=Image.new('RGB',(8*208,8*148),'#181818');d=ImageDraw.Draw(overview)
    with imageio.get_writer(str(OUT/'prediction_64.mp4'),fps=8,macro_block_size=16) as writer:
        for i,a in enumerate(t['target_angles_deg']):
            p=OUT/'raw'/f'{i:02d}.png'
            with Image.open(p) as im:
                assert im.size==(832,480)
                panel=Image.new('RGB',(832,512),'#181818');panel.paste(im,(0,32))
                ImageDraw.Draw(panel).text((12,10),f'360 orbit | frame {i:02d}/63 | {a:.2f} deg | batch {i//8+1}/8',fill='white')
                writer.append_data(np.array(panel))
                x=i%8*208;y=i//8*148;overview.paste(im.resize((208,120),Image.Resampling.LANCZOS),(x,y+28))
                d.text((x+5,y+8),f'{i:02d} | {a:.1f} deg',fill='white')
            records.append(dict(index=i,angle_deg=a,path=str(p),sha256=e.digest(p)))
    overview.save(OUT/'overview_64.jpg',quality=95)
    indices=[0,7,15,16,23,31,32,39,47,48,55,63]
    e.contact([OUT/'raw'/f'{i:02d}.png' for i in indices],[f'{i:02d}: {t["target_angles_deg"][i]:.2f} deg' for i in indices],OUT/'keyframes.jpg',columns=3)
    seams=[62,63,0,1]+[i for j in range(8,64,8) for i in [j-2,j-1,j,j+1]]
    e.contact([OUT/'raw'/f'{i:02d}.png' for i in seams],[f'{i:02d}: batch {i//8+1}' for i in seams],OUT/'batch_boundaries.jpg',columns=4)
    import subprocess
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,width,height,r_frame_rate,duration','-of','json',str(OUT/'prediction_64.mp4')]))
    assert int(probe['streams'][0]['nb_read_frames'])==64
    e.validate_manifest(read(OUT/'inference_manifest.json'))
    e.validate_manifest(read(PARENT/'inference_manifest.json'))
    e.save(OUT/'verification.json',dict(frame_count=64,unique_images=len({r['sha256'] for r in records}),frames=records,video=probe,source_and_trajectory_hashes_verified=True,visual_review='pending'))
    e.save(OUT/'status.json',dict(state='completed_pending_visual_review',completed_frames=64,target_count=64))


def main():
    import torch
    from vggtdiff import VGGTDiff
    t,manifest=prepare();e.validate_manifest(manifest)
    (OUT/'raw').mkdir(exist_ok=True);(OUT/'batches').mkdir(exist_ok=True)
    parent=read(PARENT/'inference_manifest.json')
    for w in parent['weights']:
        stat=Path(w['path']).stat();assert stat.st_size==w['size'] and stat.st_mtime_ns==w['mtime_ns']
    pending=[]
    for b in range(8):
        result=OUT/'batches'/f'{b:02d}.json'
        if result.exists():
            for r in read(result)['frames']:assert e.digest(Path(r['path']))==r['sha256']
        else:pending.append(b)
    if pending:
        e.save(OUT/'status.json',dict(state='loading',completed_frames=(8-len(pending))*8,target_count=64))
        model=VGGTDiff(parent['checkpoint'],parent['omega'],base_model_dir=parent['base'],device='cuda',vram_limit_gib=40,training_mode='nvs',load_full_checkpoint_directly=True)
        assert not model.pipe.dit.training
        images=[Image.open(OUT/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
        for b in pending:
            indices=list(range(b*8,(b+1)*8))
            w=np.array(t['source_w2c']+[t['target_w2c'][i] for i in indices],dtype=np.float32)
            k=np.array(t['source_intrinsics']+[t['target_intrinsics'][i] for i in indices],dtype=np.float32)
            assert w.shape==(14,4,4) and k.shape==(14,3,3)
            done=len(list((OUT/'batches').glob('*.json')))*8
            e.save(OUT/'status.json',dict(state='generating',batch=b+1,completed_frames=done,target_count=64))
            print(f'START_BATCH {b+1}/8',flush=True);start=time.time();torch.cuda.reset_peak_memory_stats()
            outputs=model.generate(images,w,k,height=480,width=832,steps=50,seed=20261020)
            assert len(outputs)==8
            records=[]
            for i,im in zip(indices,outputs):
                assert im.size==(832,480)
                p=OUT/'raw'/f'{i:02d}.png';im.save(p);records.append(dict(index=i,path=str(p),sha256=e.digest(p)))
            result=dict(batch=b+1,seed=20261020,steps=50,frames=records,elapsed_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            e.save(OUT/'batches'/f'{b:02d}.json',result)
            e.contact([Path(r['path']) for r in records],[f'{i:02d}: {t["target_angles_deg"][i]:.2f}' for i in indices],OUT/'batches'/f'{b:02d}.jpg',columns=4)
            print(f'COMPLETED_BATCH {b+1}/8 elapsed={result["elapsed_seconds"]:.1f}s',flush=True)
    assemble();print('COMPLETED_ALL_64',flush=True)


if __name__=='__main__':main()
