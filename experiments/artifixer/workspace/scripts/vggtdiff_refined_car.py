"""Render and freeze a refined car, select references, infer all 64 cameras."""
from __future__ import annotations
from experiment_paths import checkpoint_path, omega_path, base_model_path

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vggtdiff_white_car import BASE, SOURCE, REPO, digest, save, sheet

PLY = BASE / 'artifixer3d/H/pred_gs_5k/refined_local.ply'
OUT = BASE / 'vggtdiff_refined_H5k_nvs64_v1'
TRANSFORMS = SOURCE / 'transforms.json'


def render():
    import torch
    from worldact.data.gaussians import GaussianScene
    from worldact.data.poses import CameraPose

    if (OUT/'render_manifest.json').exists():
        raise FileExistsError('Rendered inputs already frozen')
    tf=json.loads(TRANSFORMS.read_text())
    assert len(tf['frames'])==64 and 'applied_transform' not in tf
    for name in ['baseline/rgb','baseline/alpha']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    scene=GaussianScene.from_ply(PLY);scene.params=scene.params.to('cuda')
    flip=np.diag([1.,-1.,-1.,1.]);records=[];boxes=[];cameras=[]
    roundtrip=0.
    for i,f in enumerate(tf['frames']):
        assert (int(f.get('w',tf['w'])),int(f.get('h',tf['h'])))==(512,512)
        k=np.array([[f.get('fl_x',tf['fl_x']),0,f.get('cx',tf['cx'])],
                    [0,f.get('fl_y',tf['fl_y']),f.get('cy',tf['cy'])],[0,0,1.]])
        w=np.linalg.inv(np.asarray(f['transform_matrix'])@flip)
        roundtrip=max(roundtrip,float(np.abs(np.linalg.inv(w)@flip-np.asarray(f['transform_matrix'])).max()))
        assert np.allclose(np.linalg.det(w[:3,:3]),1,atol=1e-5)
        camera=CameraPose(extrinsic=torch.tensor(w,dtype=torch.float32),
            intrinsic=torch.tensor(k,dtype=torch.float32),width=512,height=512)
        with torch.inference_mode():
            view=scene.render(camera.to('cuda'))
            alpha=view.alpha.float().clamp(0,1)
            rgb=(view.rgb.float()-(1-alpha).unsqueeze(-1)).clamp(0,1)
        p=OUT/'baseline/rgb'/f'{i:05d}.png';a=OUT/'baseline/alpha'/p.name
        Image.fromarray((rgb.cpu().numpy()*255).round().astype('uint8')).save(p)
        Image.fromarray((alpha.cpu().numpy()*255).round().astype('uint8')).save(a)
        yy,xx=np.where(alpha.cpu().numpy()>13/255)
        assert len(xx)>100,'Object not visible; verify coordinate system'
        box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        boxes.append(box)
        cameras.append(dict(w2c=w.tolist(),intrinsics=k.tolist()))
        records.append(dict(index=i,rgb_sha256=digest(p),alpha_sha256=digest(a),support_box=box))
        print('RENDERED',i,flush=True)
    assert roundtrip<1e-5
    save(OUT/'render_manifest.json',dict(ply=str(PLY),ply_sha256=digest(PLY),
        transforms=str(TRANSFORMS),transforms_sha256=digest(TRANSFORMS),frames=records,
        cameras=cameras,pose_roundtrip_error=roundtrip,gaussians=int(scene.params.means.shape[0]),
        support_union=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]))
    sheet([OUT/'baseline/rgb'/f'{i:05d}.png' for i in range(64)],
          [f'H5k {i:02d}' for i in range(64)],OUT/'baseline_contact.jpg',8,192)


def validate(m):
    for key in ['ply','transforms']:
        assert digest(Path(m[key]))==m[key+'_sha256'],f'{key} changed'
    for f in m['frames']:
        for kind in ['rgb','alpha']:
            assert digest(OUT/f'baseline/{kind}'/f"{f['index']:05d}.png")==f[kind+'_sha256']
    for f in m.get('frozen_files',[]):
        assert digest(OUT/f['path'])==f['sha256'],f['path']


def freeze(args):
    if (OUT/'manifest.json').exists():raise FileExistsError('Already frozen')
    m=json.loads((OUT/'render_manifest.json').read_text());validate(m)
    refs=args.refs;crop=args.crop
    assert len(refs)==len(set(refs))==6 and all(0<=i<64 for i in refs)
    x0,y0,x1,y1=crop
    assert -32<=x0<x1<=544 and -32<=y0<y1<=544
    assert abs((x1-x0)/(y1-y0)-832/480)<1e-10,'Crop must preserve aspect'
    # This distilled PLY has low-opacity floaters across the entire canvas.
    # Preserve all alpha>0.25 support, while documenting discarded faint pixels.
    solid_boxes=[];outside=[]
    for i in range(64):
        alpha=np.asarray(Image.open(OUT/'baseline/alpha'/f'{i:05d}.png'))
        y,x=np.where(alpha>64)
        solid_boxes.append([int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])
        valid=np.zeros((512,512),dtype=bool)
        valid[max(y0,0):min(y1,512),max(x0,0):min(x1,512)]=True
        outside.append(float(alpha[~valid].sum()/max(alpha.sum(),1)))
    box=[min(b[0] for b in solid_boxes),min(b[1] for b in solid_boxes),
         max(b[2] for b in solid_boxes),max(b[3] for b in solid_boxes)]
    assert x0<=box[0] and y0<=box[1] and x1>=box[2] and y1>=box[3]
    s=832/(x1-x0);a=np.array([[s,0,-x0*s],[0,s,-y0*s],[0,0,1.]])
    (OUT/'source_views').mkdir(exist_ok=True);(OUT/'baseline/model').mkdir(exist_ok=True)
    frozen=[];err=0.;ks=[]
    for i,c in enumerate(m['cameras']):
        k=np.asarray(c['intrinsics']);kp=a@k;ks.append(kp.tolist())
        p=np.array([[0,0,1],[255.5,255.5,1],[511,511,1.]]).T
        err=max(err,float(np.abs(np.linalg.inv(kp)@a@p-np.linalg.inv(k)@p).max()))
        with Image.open(OUT/'baseline/rgb'/f'{i:05d}.png') as im:
            processed=im.crop(crop).resize((832,480),Image.Resampling.BICUBIC)
        processed.save(OUT/'baseline/model'/f'{i:05d}.png')
        if i in refs:
            dest=OUT/'source_views'/f'{refs.index(i):02d}.png';processed.save(dest)
            frozen.append(dict(path=str(dest.relative_to(OUT)),sha256=digest(dest)))
    assert err<1e-10
    trajectory=dict(source_w2c=[m['cameras'][i]['w2c'] for i in refs],
        target_w2c=[c['w2c'] for c in m['cameras']],source_intrinsics=[ks[i] for i in refs],
        target_intrinsics=ks,camera_convention='OpenCV W2C',intrinsics_image_size=[832,480])
    save(OUT/'trajectory.json',trajectory)
    frozen.append(dict(path='trajectory.json',sha256=digest(OUT/'trajectory.json')))
    m.update(source_indices=refs,crop=crop,pixel_transform=a.tolist(),ray_mapping_error=err,
        frozen_files=frozen,solid_support_union_alpha_gt_64=box,
        alpha_mass_outside_crop_by_frame=outside,
        source_selection='Visually selected from this input PLY; not complete-car GT',
        checkpoint=str(checkpoint_path()),
        omega=str(omega_path()),
        base=str(base_model_path()))
    save(OUT/'manifest.json',m)
    mask=Image.new('L',(512,512));ImageDraw.Draw(mask).rectangle((x0,y0,x1-1,y1-1),fill=255)
    mask.save(OUT/'prediction_validity.png')
    sheet([OUT/'baseline/rgb'/f'{i:05d}.png' for i in refs],
          [f'reference {i:02d}' for i in refs],OUT/'source_contact.jpg',3,320)
    save(OUT/'status.json',dict(state='prepared',completed_batches=[],completed_frames=0))


def review():
    import imageio.v2 as imageio
    run=OUT/'full';ids=sorted(int(p.stem) for p in (run/'restored_512').glob('*.png'))
    if not ids:return
    paths=[];labels=[];video=[]
    for i in ids:
        before=OUT/'baseline/rgb'/f'{i:05d}.png';after=run/'restored_512'/before.name
        paths.extend([before,after]);labels.extend([f'H5k input {i:02d}',f'VGGT-Diff {i:02d}'])
        panel=Image.new('RGB',(1024,544),'#181818');draw=ImageDraw.Draw(panel)
        for x,p,title in [(0,before,'H5k input PLY'),(512,after,'VGGT-Diff')]:
            with Image.open(p) as im:panel.paste(im,(x,32))
            draw.text((x+8,8),f'{title} | frame {i:02d}',fill='white')
        video.append(np.asarray(panel))
    sheet(paths,labels,run/'comparison.jpg',4,224)
    imageio.mimsave(run/'comparison.mp4',video,fps=12,macro_block_size=16)
    sheet([run/'restored_512'/f'{i:05d}.png' for i in ids],
          [f'prediction {i:02d}' for i in ids],run/'predictions_contact.jpg',8,192)


def infer(args):
    import torch
    from vggtdiff import VGGTDiff
    m=json.loads((OUT/'manifest.json').read_text());validate(m)
    config=dict(manifest_sha256=digest(OUT/'manifest.json'),steps=50,seed=20260823,
        batch_size=8,targets=list(range(64)),source_indices=m['source_indices'],dit_eval=True,
        resolution=[832,480],vram_limit_gib=20,training_mode='nvs',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device_name=torch.cuda.get_device_name())
    run=OUT/'full';run.mkdir(exist_ok=True)
    if (run/'settings.json').exists():assert json.loads((run/'settings.json').read_text())==config
    save(run/'settings.json',config)
    for folder in ['raw','restored_512']:(run/folder).mkdir(exist_ok=True)
    status=dict(state='loading',completed_batches=[],completed_frames=0)
    save(OUT/'status.json',status)
    model=VGGTDiff(m['checkpoint'],m['omega'],base_model_dir=m['base'],device='cuda',
        vram_limit_gib=20,training_mode='nvs',load_full_checkpoint_directly=True)
    assert not model.pipe.dit.training
    images=[Image.open(OUT/'source_views'/f'{j:02d}.png').convert('RGB') for j in range(6)]
    t=json.loads((OUT/'trajectory.json').read_text())
    # First batch spans the orbit for early inspection; all 64 targets generated exactly once.
    first=[4,12,20,28,36,44,52,60]
    rest=[i for i in range(64) if i not in first]
    batches=[first]+[rest[i:i+8] for i in range(0,len(rest),8)]
    assert sorted(sum(batches,[]))==list(range(64))
    save(run/'batches.json',batches)
    for b,ids in enumerate(batches):
        done=run/f'batch_{b:02d}.json'
        if done.exists():
            record=json.loads(done.read_text());assert record['indices']==ids
            for item in record['frames']:
                assert digest(run/'raw'/f"{item['frame']:05d}.png")==item['sha256']
            status['completed_batches'].append(b);status['completed_frames']+=len(ids)
            continue
        status.update(state='running',active_batch=b,active_targets=ids);save(OUT/'status.json',status)
        print('START_BATCH',b,ids,flush=True)
        w=np.asarray(t['source_w2c']+[t['target_w2c'][i] for i in ids],dtype=np.float32)
        k=np.asarray(t['source_intrinsics']+[t['target_intrinsics'][i] for i in ids],dtype=np.float32)
        torch.cuda.reset_peak_memory_stats();start=time.time()
        outputs=model.generate(images,w,k,height=480,width=832,steps=50,seed=20260823+b)
        assert len(outputs)==8
        records=[];x0,y0,x1,y1=m['crop']
        for i,im in zip(ids,outputs):
            assert im.size==(832,480)
            dest=run/'raw'/f'{i:05d}.png';im.save(dest)
            canvas=Image.new('RGB',(512,512),'black')
            canvas.paste(im.resize((x1-x0,y1-y0),Image.Resampling.BICUBIC),(x0,y0))
            canvas.save(run/'restored_512'/dest.name)
            records.append(dict(frame=i,sha256=digest(dest)))
        save(done,dict(indices=ids,seed=20260823+b,elapsed_seconds=time.time()-start,
            peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,frames=records))
        status['completed_batches'].append(b);status['completed_frames']+=len(ids)
        save(OUT/'status.json',status);review()
        print('COMPLETED_BATCH',b,flush=True)
    validate(m)
    assert len(list((run/'raw').glob('*.png')))==64
    status.update(state='completed_pending_visual_review',active_batch=None,active_targets=[])
    save(OUT/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['render','freeze','infer','review'])
    p.add_argument('--refs',type=int,nargs=6)
    p.add_argument('--crop',type=int,nargs=4)
    a=p.parse_args()
    try:
        {'render':render,'freeze':lambda:freeze(a),'infer':lambda:infer(a),'review':review}[a.action]()
    except Exception as e:
        if a.action=='infer':save(OUT/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
