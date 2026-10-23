"""Four controlled damaged-car NVS groups; one shared model, frozen inputs."""
from __future__ import annotations
from experiment_paths import checkpoint_path, omega_path, base_model_path

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vggtdiff_white_car import OUT as ORIGINAL, PILOT, REFS, SOURCE, REPO, digest, save, sheet

OUT=ORIGINAL.parent/'vggtdiff_factorial_v2'
GOOD=[0,2,6,10,14,62]
CROP=(9,190,503,475)
GROUPS={'A':(REFS,'wide'),'B':(GOOD,'wide'),'C':(REFS,'crop'),'D':(GOOD,'crop')}


def pixel_transform(viewport):
    if viewport=='wide':
        return np.array([[480/512,0,176],[0,480/512,0],[0,0,1.]])
    if viewport!='crop':raise ValueError(viewport)
    x0,y0,x1,y1=CROP
    sx,sy=832/(x1-x0),480/(y1-y0)
    assert abs(sx-sy)<1e-12
    return np.array([[sx,0,-sx*x0],[0,sy,-sy*y0],[0,0,1.]])


def transform_image(image,viewport):
    if viewport=='wide':
        canvas=Image.new('RGB',(832,480),'black')
        canvas.paste(image.resize((480,480),Image.Resampling.BICUBIC),(176,0))
        return canvas
    if viewport!='crop':raise ValueError(viewport)
    return image.crop(CROP).resize((832,480),Image.Resampling.BICUBIC)


def restore_image(image,viewport):
    if viewport=='wide':
        return image.crop((176,0,656,480)).resize((512,512),Image.Resampling.BICUBIC)
    if viewport!='crop':raise ValueError(viewport)
    canvas=Image.new('RGB',(512,512),'black')
    x0,y0,x1,y1=CROP
    canvas.paste(image.resize((x1-x0,y1-y0),Image.Resampling.BICUBIC),(x0,y0))
    return canvas


def validate_frozen(manifest):
    for name,entry in manifest['frozen_files'].items():
        if digest(Path(entry['path']))!=entry['sha256']:
            raise ValueError(f'Frozen input changed: {name}')


def prepare():
    if (OUT/'manifest.json').exists():raise FileExistsError('Experiment already frozen')
    old=json.loads((ORIGINAL/'manifest.json').read_text())
    tf=json.loads(Path(old['transforms']).read_text())
    trajectory=json.loads((ORIGINAL/'trajectory.json').read_text())
    assert len(tf['frames'])==64 and len(set(PILOT))==8
    assert not set(GOOD)&set(PILOT) and not set(REFS)&set(PILOT)
    frozen={}
    for name,path in [('ply',Path(old['ply'])),('transforms',Path(old['transforms']))]:
        checksum=digest(path)
        assert checksum==old[name+'_sha256']
        frozen[name]={'path':str(path),'sha256':checksum}
    boxes=[]
    for i in range(64):
        rgb=ORIGINAL/'baseline/rgb'/f'{i:05d}.png'
        alpha=ORIGINAL/'baseline/alpha'/rgb.name
        assert digest(rgb)==old['frames'][i]['rgb_sha256']
        for kind,path in [('rgb',rgb),('alpha',alpha)]:
            frozen[f'{kind}_{i:02d}']={'path':str(path),'sha256':digest(path)}
        a=np.asarray(Image.open(alpha))>13
        y,x=np.where(a);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        assert box[0]>=CROP[0] and box[1]>=CROP[1] and box[2]<=CROP[2] and box[3]<=CROP[3]
        boxes.append(box)
    groups={}
    for name,(refs,viewport) in GROUPS.items():
        dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
        (dest/'source_views').mkdir(exist_ok=True)
        (dest/'baseline_model').mkdir(exist_ok=True)
        a=pixel_transform(viewport);ks=[];rays_error=0.;occupancy=[]
        for i,frame in enumerate(tf['frames']):
            k=np.array([[frame.get('fl_x',tf['fl_x']),0,frame.get('cx',tf['cx'])],
                        [0,frame.get('fl_y',tf['fl_y']),frame.get('cy',tf['cy'])],[0,0,1.]])
            ka=a@k;ks.append(ka.tolist())
            pixels=np.array([[31.5,214.5,1],[255.5,332.5,1],[479.5,449.5,1]]).T
            rays_error=max(rays_error,float(np.abs(np.linalg.inv(ka)@a@pixels-np.linalg.inv(k)@pixels).max()))
            if i in set(refs)|set(PILOT):
                with Image.open(ORIGINAL/'baseline/rgb'/f'{i:05d}.png') as im:
                    processed=transform_image(im.convert('RGB'),viewport)
                processed.save(dest/'baseline_model'/f'{i:05d}.png')
                if i in refs:
                    path=dest/'source_views'/f'{refs.index(i):02d}.png';processed.save(path)
                    frozen[f'{name}_source_{refs.index(i)}']={'path':str(path),'sha256':digest(path)}
                    with Image.open(ORIGINAL/'baseline/alpha'/f'{i:05d}.png') as mask:
                        m=mask.crop(CROP) if viewport=='crop' else mask
                        fraction=float((np.array(m)>13).mean())
                        occupancy.append(fraction if viewport=='crop' else fraction*480/832)
        assert rays_error<1e-10
        traj=dict(source_w2c=[trajectory['target_w2c'][i] for i in refs],
                  target_w2c=[trajectory['target_w2c'][i] for i in PILOT],
                  source_intrinsics=[ks[i] for i in refs],target_intrinsics=[ks[i] for i in PILOT],
                  intrinsics_image_size=[832,480],target_indices=PILOT)
        save(dest/'trajectory.json',traj)
        frozen[f'{name}_trajectory']={'path':str(dest/'trajectory.json'),'sha256':digest(dest/'trajectory.json')}
        groups[name]={'source_indices':refs,'viewport':viewport,'pixel_transform':a.tolist(),
                      'ray_mapping_error':rays_error,'foreground_fractions':occupancy}
        sheet([dest/'source_views'/f'{i:02d}.png' for i in range(6)],
              [f'{name} source {i:02d}' for i in refs],dest/'source_contact.jpg',3,320)
        valid=Image.new('L',(512,512),0)
        ImageDraw.Draw(valid).rectangle((CROP[0],CROP[1],CROP[2]-1,CROP[3]-1) if viewport=='crop' else (0,0,511,511),fill=255)
        valid.save(dest/'prediction_validity.png')
    manifest={'groups':groups,'targets':PILOT,'steps':50,'seed':20260823,'resolution':[832,480],
              'crop':list(CROP),'alpha_threshold':13/255,'all64_support_boxes':boxes,
              'frozen_files':frozen,'weight_checksums':json.loads((ORIGINAL/'weight_checksums.json').read_text()),
              'reliable_reference_review':'0/2/6/10/14/62 inspected; minor artifacts remain; front/left coverage only',
              'limitations':['reference-quality change also changes camera coverage','crop predictions do not cover pixels outside prediction_validity.png','no complete-vehicle ground truth']}
    save(OUT/'manifest.json',manifest)
    save(OUT/'status.json',{'state':'prepared','groups':{g:'pending' for g in GROUPS}})
    (OUT/'loader.patch').write_text(subprocess.check_output(['git','-C',str(REPO/'thirdparty/VGGT-Diff'),'diff','--','vggtdiff/model.py'],text=True))
    print('PREPARED',OUT,groups,flush=True)


def review(seed=20260823):
    import imageio.v2 as imageio
    for group in GROUPS:
        run=OUT/group/f'seed_{seed}'
        if not all((run/'raw'/f'{i:05d}.png').exists() for i in PILOT):continue
        paths=[];labels=[];video=[]
        for i in PILOT:
            before=ORIGINAL/'baseline/rgb'/f'{i:05d}.png';after=run/'restored_512'/before.name
            paths.extend([before,after]);labels.extend([f'original {i:02d}',f'{group} {i:02d}'])
            panel=Image.new('RGB',(1024,544),'#181818');draw=ImageDraw.Draw(panel)
            for x,p,label in [(0,before,'original PLY'),(512,after,group)]:
                with Image.open(p) as im:panel.paste(im,(x,32))
                draw.text((x+8,8),f'{label} frame {i:02d}',fill='white')
            video.append(np.array(panel))
        sheet(paths,labels,run/'comparison.jpg',4)
        imageio.mimsave(run/'comparison.mp4',video,fps=2,macro_block_size=16)
    if not all((OUT/g/f'seed_{seed}/raw/{i:05d}.png').exists() for g in GROUPS for i in PILOT):return
    for ids,label in [(PILOT,'all_views'),([20,28,44,52],'key_views')]:
        paths=[];labels=[]
        for i in ids:
            paths.append(ORIGINAL/'baseline/rgb'/f'{i:05d}.png');labels.append(f'original {i:02d}')
            for g in GROUPS:
                paths.append(OUT/g/f'seed_{seed}/restored_512/{i:05d}.png');labels.append(f'{g} frame {i:02d}')
        sheet(paths,labels,OUT/f'{label}_seed_{seed}.jpg',5,256 if label=='key_views' else 224)
    for i in PILOT:
        paths=[ORIGINAL/'baseline/rgb'/f'{i:05d}.png']+[OUT/g/f'seed_{seed}/restored_512/{i:05d}.png' for g in GROUPS]
        sheet(paths,['original']+list(GROUPS),OUT/f'frame_{i:02d}_seed_{seed}.jpg',5,512)


def run(args):
    import torch
    from vggtdiff import VGGTDiff
    manifest=json.loads((OUT/'manifest.json').read_text());validate_frozen(manifest)
    status=json.loads((OUT/'status.json').read_text())
    assert args.steps==manifest['steps']
    model=VGGTDiff(str(checkpoint_path()),
        str(omega_path()),
        base_model_dir=str(base_model_path()),
        device='cuda',vram_limit_gib=20,training_mode='nvs',load_full_checkpoint_directly=True)
    assert not model.pipe.dit.training
    for group in args.groups:
        data=manifest['groups'][group];dest=OUT/group;run=dest/f'seed_{args.seed}'
        run.mkdir(exist_ok=True)
        config=dict(group=group,seed=args.seed,steps=args.steps,source_indices=data['source_indices'],
            targets=PILOT,viewport=data['viewport'],pixel_transform=data['pixel_transform'],
            resolution=[832,480],training_mode='nvs',dit_eval=True,vram_limit_gib=20,
            device_name=torch.cuda.get_device_name(),cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
            manifest_sha256=digest(OUT/'manifest.json'))
        if (run/'settings.json').exists():assert json.loads((run/'settings.json').read_text())==config
        save(run/'settings.json',config)
        if all((run/'raw'/f'{i:05d}.png').exists() for i in PILOT):
            print('SKIP',group,flush=True);continue
        for folder in ['raw','restored_512']:(run/folder).mkdir(exist_ok=True)
        images=[Image.open(dest/'source_views'/f'{j:02d}.png').convert('RGB') for j in range(6)]
        t=json.loads((dest/'trajectory.json').read_text())
        w=np.asarray(t['source_w2c']+t['target_w2c'],dtype=np.float32)
        k=np.asarray(t['source_intrinsics']+t['target_intrinsics'],dtype=np.float32)
        status.update(state='running',active_group=group,seed=args.seed)
        status['groups'][group]='running';save(OUT/'status.json',status)
        print('START_GROUP',group,'SEED',args.seed,flush=True)
        torch.cuda.reset_peak_memory_stats();start=time.time()
        outputs=model.generate(images,w,k,height=480,width=832,steps=args.steps,seed=args.seed)
        assert len(outputs)==8
        records=[]
        for i,im in zip(PILOT,outputs):
            assert im.size==(832,480)
            raw=run/'raw'/f'{i:05d}.png';im.save(raw)
            restore_image(im,data['viewport']).save(run/'restored_512'/raw.name)
            records.append({'frame':i,'sha256':digest(raw)})
        save(run/'result.json',dict(elapsed_seconds=time.time()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,frames=records,visual_review='pending'))
        status['groups'][group]='completed';save(OUT/'status.json',status)
        print('COMPLETED_GROUP',group,flush=True)
        review(args.seed)
    validate_frozen(manifest)
    status.update(state='generation_completed_pending_review',active_group=None);save(OUT/'status.json',status)
    review(args.seed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','review'])
    p.add_argument('--groups',nargs='+',choices=list(GROUPS),default=list(GROUPS))
    p.add_argument('--seed',type=int,default=20260823)
    p.add_argument('--steps',type=int,default=50)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args)
    else:review(args.seed)
