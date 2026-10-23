"""Frozen-PLY white-car NVS experiment: prepare, infer, review (no 3D distillation)."""
from __future__ import annotations
from experiment_paths import checkpoint_path, omega_path, base_model_path

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'worldact')]
BASE = REPO / 'pipeline_exp/realworld/scene-1-nofull/06_worldact_white_car_v1/artifixer_episode_chain_64x4'
SOURCE = BASE / 'source_64/realworld_white_car_cleanup_64_start19_front_views'
OUT = BASE / 'vggtdiff_nvs64_v1'
REFS = [0, 11, 21, 32, 43, 53]
PILOT = [4, 12, 20, 28, 36, 44, 52, 60]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sheet(paths, labels, destination, columns=4, size=256):
    canvas = Image.new('RGB', (columns * size, ((len(paths)+columns-1)//columns)*(size+28)), '#181818')
    draw = ImageDraw.Draw(canvas)
    for j, (path, label) in enumerate(zip(paths, labels)):
        x, y = (j % columns)*size, (j // columns)*(size+28)
        with Image.open(path) as image:
            canvas.paste(image.convert('RGB').resize((size,size)), (x,y+28))
        draw.text((x+6,y+6), label, fill='white')
    canvas.save(destination, quality=94)


def prepare(args):
    import torch
    from worldact.data.gaussians import GaussianScene
    from worldact.data.poses import CameraPose

    out = args.output
    if (out / 'manifest.json').exists():
        raise FileExistsError('Prepared experiment already exists; use infer/review to resume.')
    for name in ['baseline/rgb', 'baseline/alpha', 'source_views', 'baseline/letterboxed']:
        (out / name).mkdir(parents=True, exist_ok=True)
    tf = json.loads((SOURCE / 'transforms.json').read_text())
    assert len(tf['frames']) == 64 and 'applied_transform' not in tf
    ply = SOURCE / 'initial_object/object_local.ply'
    scene = GaussianScene.from_ply(ply)
    scene.params = scene.params.to(args.device)
    w2cs, intrinsics, records = [], [], []
    flip = np.diag([1., -1., -1., 1.])
    # Preserve the square view; add black side margins, never stretch/crop the car.
    pixel_map = np.array([[480/512, 0, 176], [0, 480/512, 0], [0, 0, 1.]])
    max_roundtrip, max_projection = 0., 0.
    for i, frame in enumerate(tf['frames']):
        w, h = int(frame.get('w', tf['w'])), int(frame.get('h', tf['h']))
        assert (w,h) == (512,512)
        k = np.array([[frame.get('fl_x',tf['fl_x']),0,frame.get('cx',tf['cx'])],
                      [0,frame.get('fl_y',tf['fl_y']),frame.get('cy',tf['cy'])],[0,0,1.]])
        c2w = np.asarray(frame['transform_matrix'], dtype=np.float64) @ flip
        w2c = np.linalg.inv(c2w)
        max_roundtrip = max(max_roundtrip, float(np.abs(np.linalg.inv(w2c) @ flip - frame['transform_matrix']).max()))
        assert np.allclose(np.linalg.det(c2w[:3,:3]), 1, atol=1e-5)
        kp = pixel_map @ k
        # Verify that padding/scaling preserves rays at sampled original pixels.
        pixels = np.array([[0.,0.,1.],[255.5,255.5,1.],[511.,511.,1.]]).T
        max_projection = max(max_projection,float(np.abs(np.linalg.inv(kp) @ pixel_map @ pixels - np.linalg.inv(k) @ pixels).max()))
        cam = CameraPose(extrinsic=torch.tensor(w2c,dtype=torch.float32),
                         intrinsic=torch.tensor(k,dtype=torch.float32),width=w,height=h)
        with torch.inference_mode():
            view = scene.render(cam.to(args.device))
            alpha = view.alpha.float().clamp(0,1)
            # WorldAct's renderer uses a white background; remove its contribution.
            rgb = (view.rgb.float() - (1-alpha).unsqueeze(-1)).clamp(0,1)
        image = Image.fromarray((rgb.cpu().numpy()*255).round().astype('uint8'))
        path = out / 'baseline/rgb' / f'{i:05d}.png'
        image.save(path)
        Image.fromarray((alpha.cpu().numpy()*255).round().astype('uint8')).save(out/'baseline/alpha'/path.name)
        padded = Image.new('RGB',(832,480),'black')
        padded.paste(image.resize((480,480),Image.Resampling.BICUBIC),(176,0))
        padded.save(out/'baseline/letterboxed'/path.name)
        if i in REFS:
            padded.save(out/'source_views'/f'{REFS.index(i):02d}.png')
        w2cs.append(w2c.tolist());intrinsics.append(kp.tolist())
        records.append({'index':i,'rgb_sha256':digest(path),'visible_fraction':float((alpha>.01).float().mean())})
        print('rendered',i,flush=True)
    assert max_roundtrip < 1e-5 and max_projection < 1e-10
    trajectory = dict(source_w2c=[w2cs[i] for i in REFS],target_w2c=w2cs,
                      source_intrinsics=[intrinsics[i] for i in REFS],target_intrinsics=intrinsics,
                      intrinsics_image_size=[832,480],camera_convention='OpenCV W2C')
    save(out/'trajectory.json',trajectory)
    save(out/'manifest.json',dict(ply=str(ply),ply_sha256=digest(ply),
         transforms=str(SOURCE/'transforms.json'),transforms_sha256=digest(SOURCE/'transforms.json'),
         source_indices=REFS,pilot_indices=PILOT,frames=records,resolution=[832,480],
         pixel_map=pixel_map.tolist(),crop_to_square=[176,0,656,480],
         pose_roundtrip_error=max_roundtrip,ray_mapping_error=max_projection,
         vggtdiff_commit=subprocess.check_output(['git','-C',str(REPO/'thirdparty/VGGT-Diff'),'rev-parse','HEAD'],text=True).strip(),
         limitation='Completion hypotheses, no complete-vehicle GT; no repaired image conditioning.'))
    sheet([out/'baseline/rgb'/f'{i:05d}.png' for i in REFS], [f'source {i:02d}' for i in REFS],out/'source_contact.jpg',3)
    sheet([out/'baseline/rgb'/f'{i:05d}.png' for i in range(64)], [f'original {i:02d}' for i in range(64)],out/'baseline_contact.jpg',8,160)
    print('PREPARED',out,flush=True)


def infer(args):
    import torch
    from vggtdiff import VGGTDiff

    out = args.output
    manifest = json.loads((out/'manifest.json').read_text())
    assert digest(Path(manifest['ply'])) == manifest['ply_sha256']
    assert digest(Path(manifest['transforms'])) == manifest['transforms_sha256']
    config = json.loads(args.checkpoint.with_name('config.json').read_text())
    assert config['model_family'] == 'nvs'
    targets = PILOT if args.stage == 'pilot' else list(range(64))
    batches = [targets[i:i+8] for i in range(0,len(targets),8)]
    run = out/args.stage
    run.mkdir(exist_ok=True)
    for name in ['raw','square_512']:
        (run/name).mkdir(exist_ok=True)
    save(run/'settings.json',dict(checkpoint=str(args.checkpoint),omega=str(args.omega),base=str(args.base),
         stage=args.stage,targets=targets,batches=batches,steps=args.steps,seed=args.seed,
         vram_limit_gib=args.vram_limit_gib,source_indices=REFS,training_mode='nvs'))
    trajectory = json.loads((out/'trajectory.json').read_text())
    images = [Image.open(out/'source_views'/f'{i:02d}.png').convert('RGB') for i in range(6)]
    model = VGGTDiff(checkpoint=str(args.checkpoint),omega_checkpoint=str(args.omega),base_model_dir=str(args.base),
                     device=args.device,vram_limit_gib=args.vram_limit_gib,training_mode='nvs',
                     load_full_checkpoint_directly=True)
    for b, indices in enumerate(batches):
        if all((run/'raw'/f'{i:05d}.png').exists() for i in indices):
            print('skip completed batch',indices,flush=True);continue
        w2c = np.array(trajectory['source_w2c']+[trajectory['target_w2c'][i] for i in indices],dtype=np.float32)
        k = np.array(trajectory['source_intrinsics']+[trajectory['target_intrinsics'][i] for i in indices],dtype=np.float32)
        start = time.time()
        predicted = model.generate(images,w2c,k,height=480,width=832,steps=args.steps,seed=args.seed+b)
        assert len(predicted)==len(indices)
        for i,image in zip(indices,predicted):
            image.save(run/'raw'/f'{i:05d}.png')
            image.crop((176,0,656,480)).resize((512,512),Image.Resampling.BICUBIC).save(run/'square_512'/f'{i:05d}.png')
        save(run/f'batch_{b:02d}.json',dict(indices=indices,seed=args.seed+b,elapsed_seconds=time.time()-start,
             peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
        print('COMPLETED',indices,flush=True)
    review(args)


def review(args):
    import imageio.v2 as imageio
    run = args.output/args.stage
    indices = sorted(int(p.stem) for p in (run/'square_512').glob('*.png'))
    assert indices
    paths,labels,video = [],[],[]
    for i in indices:
        original=args.output/'baseline/rgb'/f'{i:05d}.png';prediction=run/'square_512'/f'{i:05d}.png'
        paths.extend([original,prediction]);labels.extend([f'original {i:02d}',f'VGGT-Diff {i:02d}'])
        panel=Image.new('RGB',(1024,544),'#181818');draw=ImageDraw.Draw(panel)
        for x,p,title in [(0,original,'original PLY'),(512,prediction,'VGGT-Diff hypothesis')]:
            with Image.open(p) as im:panel.paste(im,(x,32))
            draw.text((x+10,8),f'{title} | frame {i:02d}',fill='white')
        video.append(np.array(panel))
    sheet(paths,labels,run/'comparison.jpg',4)
    imageio.mimsave(run/'comparison.mp4',video,fps=2 if args.stage=='pilot' else 12,macro_block_size=16)
    save(run/'review.json',dict(frame_indices=indices,assessment='pending visual review; no complete GT'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','infer','review'])
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--device',default='cuda')
    p.add_argument('--stage',choices=['pilot','full'],default='pilot')
    p.add_argument('--checkpoint',type=Path,default=Path(str(checkpoint_path())))
    p.add_argument('--omega',type=Path,default=Path(str(omega_path())))
    p.add_argument('--base',type=Path,default=Path(str(base_model_path())))
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=20260823)
    p.add_argument('--vram-limit-gib',type=float,default=20.)
    args=p.parse_args()
    {'prepare':prepare,'infer':infer,'review':review}[args.action](args)
