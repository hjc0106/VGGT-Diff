"""Render matched fixed-camera comparisons and audit exported Gaussian files."""
import json, math, subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from omegaconf import OmegaConf
import vggtdiff_car15_train as train
from vggtdiff_car15_prepare import OUT,BASE,read,e
from vggtdiff_white_car import save,digest
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch
from plyfile import PlyData

def load(stage):
    result=read(OUT/'training'/f'{stage}_result.json')
    conf=OmegaConf.load(OUT/'training'/f'{stage}_config.yaml')
    model=MixtureOfGaussians(conf)
    model.init_from_checkpoint(torch.load(result['checkpoint'],map_location='cuda',weights_only=False),setup_optimizer=False)
    model.eval();return model

@torch.inference_mode()
def render(model,w,k):
    k=np.array(k);r=torch.zeros((1,480,832,3),device='cuda')
    batch=Batch(r,r,torch.tensor(np.linalg.inv(w),dtype=torch.float32,device='cuda')[None],intrinsics_OpenCVPinholeCameraModelParameters=dict(focal_length=[k[0,0],k[1,1]],principal_point=[k[0,2],k[1,2]]))
    out=model(batch,train=False)
    return out['pred_rgb'][0].clamp(0,1).cpu().numpy(),out['pred_opacity'][0,:,:,0].cpu().numpy()

def write(path,rgb):
    path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(path)

def metrics(rgb,alpha,gt,mask):
    error=(rgb-gt)[mask]
    mse=float(np.mean(error**2));a=alpha>.5
    return dict(foreground_psnr=-10*math.log10(max(mse,1e-12)),foreground_mae=float(abs(error).mean()),silhouette_iou=float((a&mask).sum()/max((a|mask).sum(),1)),foreground_mean_opacity=float(alpha[mask].mean()))

def pair(a,b,path,label):
    im=Image.new('RGB',(1664,512));im.paste(Image.open(a),(0,32));im.paste(Image.open(b),(832,32));d=ImageDraw.Draw(im);d.text((10,10),'real6 baseline | '+label,fill='white');d.text((842,10),'mixed15 | '+label,fill='white');path.parent.mkdir(parents=True,exist_ok=True);im.save(path)

def main():
    torch.set_num_threads(4);root=OUT/'evaluation';root.mkdir(exist_ok=True)
    frames=read(OUT/'inputs.json')['frames'];val=read(OUT/'validation/cameras.json');print('VALIDATION_SCHEMA',type(val),flush=True)
    if isinstance(val,dict):val=val.get('frames',val.get('cameras',val))
    orbit=read(BASE/'vggtdiff_realvideo_orbit360_64_v2_batch8'/'trajectory.json')
    report=dict(backend='gsplat rasterizer with 3DGRUT Gaussian model/optimizer',stages={},validation_pose_warning='Estimated PnP poses; 16 and 23 inliers. Real validation observes the recorded side, not hidden-side ground truth.')
    for stage in ['real6','mixed15']:
        model=load(stage);rows=[]
        for row in frames:
            name=row['name'];rgb,alpha=render(model,np.array(row['w2c']),row['k']);write(root/stage/'training'/name,rgb)
            gt=np.array(Image.open(OUT/'images'/name))/255.;mask=np.array(Image.open(OUT/'masks'/name))>127
            rows.append(dict(name=name,kind=row['kind'],**metrics(rgb,alpha,gt,mask)))
        vals=[]
        for row in val:
            name=Path(row['name']).stem
            w=row.get('w2c');k=row.get('k',row.get('intrinsics'));rgb,alpha=render(model,np.array(w),k)
            write(root/stage/'validation'/f'{name}.png',rgb)
            gt=np.array(Image.open(OUT/'validation/images'/f'{name}.png'))/255.;mask=np.array(Image.open(OUT/'validation/masks'/f'{name}.png'))>127
            vals.append(dict(name=name,**metrics(rgb,alpha,gt,mask)))
        for i,(w,k) in enumerate(zip(orbit['target_w2c'],orbit['target_intrinsics'])):
            rgb,alpha=render(model,np.array(w),k);write(root/stage/'orbit'/f'{i:03d}.png',rgb)
        g=read(BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json')
        center=np.array(g['center']);up=np.array(g['up']);front=np.array(g['front']);left=np.array(g['left'])
        for angle in range(0,360,60):
            theta=np.deg2rad(angle);position=center+.3142*(np.cos(theta)*front-np.sin(theta)*left)+.10749*up
            w=e.look_at(position,center,up)
            rgb,alpha=render(model,w,orbit['target_intrinsics'][0]);write(root/stage/'elevated'/f'{angle:03d}.png',rgb)
        ply=PlyData.read(OUT/f'{stage}.ply')['vertex'];names=ply.data.dtype.names
        assert all(n in names for n in ['opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','f_dc_0'])
        assert all(np.isfinite(ply[n]).all() for n in names)
        aggregate={}
        for group,subset in [('real_training',[x for x in rows if x['kind']=='real']),('generated_training',[x for x in rows if x['kind']=='generated']),('independent_real',vals)]:
            aggregate[group]={key:float(np.mean([x[key] for x in subset])) for key in ['foreground_psnr','foreground_mae','silhouette_iou','foreground_mean_opacity']}
        report['stages'][stage]=dict(aggregate=aggregate,training_metrics=rows,independent_real_metrics=vals,gaussian_count=len(ply),ply_sha256=digest(OUT/f'{stage}.ply'))
        del model;torch.cuda.empty_cache();print('RENDER_COMPLETED',stage,flush=True)
    for i in range(64):pair(root/'real6/orbit'/f'{i:03d}.png',root/'mixed15/orbit'/f'{i:03d}.png',root/'comparison'/f'{i:03d}.png',f'angle {i*360/64:.3f} deg')
    subprocess.run(['ffmpeg','-y','-framerate','8','-i',str(root/'comparison/%03d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(root/'orbit360_comparison.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for angle in range(0,360,60):pair(root/'real6/elevated'/f'{angle:03d}.png',root/'mixed15/elevated'/f'{angle:03d}.png',root/'elevated_comparison'/f'{angle:03d}.png',f'elevated {angle} deg')
    e.contact([root/'elevated_comparison'/f'{a:03d}.png' for a in range(0,360,60)],[f'{a} deg' for a in range(0,360,60)],root/'elevated_contact.jpg',2)
    selected=list(range(0,64,8));e.contact([root/'comparison'/f'{i:03d}.png' for i in selected],[f'{i*360/64:.0f} deg' for i in selected],root/'orbit_contact.jpg',2)
    for row in val:
        name=row['name'];gt=np.array(Image.open(OUT/'validation/images'/name))/255.;mask=np.array(Image.open(OUT/'validation/masks'/name))>127
        gt=gt*mask[:,:,None];write(root/'validation_groundtruth'/name,gt)
        canvas=Image.new('RGB',(2496,512));draw=ImageDraw.Draw(canvas)
        for j,(label,path) in enumerate([('Independent real image',root/'validation_groundtruth'/name),('real6',root/'real6/validation'/name),('mixed15',root/'mixed15/validation'/name)]):
            canvas.paste(Image.open(path),(832*j,32));draw.text((832*j+8,10),label,fill='white')
        canvas.save(root/f'validation_comparison_{name}')
    e.contact([root/'comparison'/f'{i:03d}.png' for i in range(64)],[f'{i}: {i*360/64:.1f} deg' for i in range(64)],root/'all64_contact.jpg',4)
    for stage in ['real6','mixed15']:
        e.contact([root/stage/'training'/r['name'] for r in frames],[f"{r['name']} {r['kind']}" for r in frames],root/f'{stage}_training_contact.jpg',3)
    manifest=read(OUT/'experiment_manifest.json')
    checked=[dict(path=f['path'],matches=digest(Path(f['path']))==f['sha256']) for f in manifest['frozen_files']]
    assert all(x['matches'] for x in checked)
    baseline=read(OUT/'training/real6_result.json');mixed=read(OUT/'training/mixed15_result.json')
    assert baseline['warmup_sha256']==mixed['warmup_sha256']==digest(OUT/'warmup_checkpoint.pt')
    save(root/'audit.json',dict(frozen_files=checked,shared_warmup_sha256=baseline['warmup_sha256'],evaluator_sha256=digest(Path(__file__)),status='completed; visual review pending'))
    save(root/'metrics.json',report);print('EVALUATION_COMPLETED',flush=True)
if __name__=='__main__':main()
