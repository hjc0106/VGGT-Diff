"""Prepare a fresh ArtiFixer V6-A scene from the mixed15 Gaussian PLY."""
import sys,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from omegaconf import OmegaConf
from vggtdiff_car15_evaluate import render,write,MixtureOfGaussians
from vggtdiff_car15_prepare import OUT,BASE,read
from vggtdiff_white_car import save,digest
ROOT=OUT/'artifixer_2d_v1'
SCENE='vggtdiff_mixed15_orbit64_front_views'
PREP=ROOT/'prepared'/SCENE
REFS=[8,16,24,40,56]

def main():
    torch.set_num_threads(4)
    assert not (PREP/'split.json').exists(),'Prepared scene exists; refusing to overwrite'
    original=OUT/'mixed15.ply';source=BASE/'vggtdiff_realvideo_orbit360_64_v2_batch8/trajectory.json';trajectory=read(source)
    bounds=np.array(read(OUT/'initialization.json')['bounds_world']);center=bounds.mean(0);radius=float(np.linalg.norm(bounds[1]-bounds[0])/2)
    local=PREP/'initial_object/object_local.ply';local.parent.mkdir(parents=True,exist_ok=True)
    ply=PlyData.read(original);v=ply['vertex'].data
    for j,c in enumerate('xyz'):v[c]=(v[c]-center[j])/radius
    for j in range(3):v[f'scale_{j}']-=np.log(radius)
    ply.write(local)
    conf=OmegaConf.load(OUT/'training/mixed15_config.yaml');model=MixtureOfGaussians(conf);model.init_from_ply(str(local),init_model=False);model.eval()
    rec=PREP/f'recon_results/{SCENE}/reconstruction/{SCENE}/ours_0';frames=[];stats=[]
    for i,(w,k) in enumerate(zip(trajectory['target_w2c'],trajectory['target_intrinsics'])):
        c2w=np.linalg.inv(w);c2w[:3,3]=(c2w[:3,3]-center)/radius;w=np.linalg.inv(c2w);k=np.array(k)
        rgb,alpha=render(model,w,k);name=f'{i:05d}.png';write(rec/'renders'/name,rgb);(rec/'opacity').mkdir(exist_ok=True)
        Image.fromarray(np.uint8(np.clip(alpha,0,1)*255)).save(rec/'opacity'/name)
        (PREP/'images').mkdir(exist_ok=True);shutil.copyfile(rec/'renders'/name,PREP/'images'/name)
        opengl=c2w@np.diag([1,-1,-1,1]);frames.append(dict(file_path=f'images/{name}',transform_matrix=opengl.tolist(),fl_x=float(k[0,0]),fl_y=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]),w=832,h=480,camera_model='OPENCV'))
        stats.append(dict(index=i,angle=trajectory['target_angles_deg'][i],visible_fraction=float((alpha>.01).mean())))
        print('RENDER',i,flush=True)
    for offset,index in enumerate(REFS):
        slot=64+offset;name=f'{slot:05d}.png';frame=dict(frames[index]);frame['file_path']=f'images/{name}';frames.append(frame)
        for d in [PREP/'images',rec/'renders',rec/'opacity']:shutil.copyfile(d/f'{index:05d}.png',d/name)
    transforms=dict(camera_model='OPENCV',w=832,h=480,fl_x=frames[0]['fl_x'],fl_y=frames[0]['fl_y'],cx=frames[0]['cx'],cy=frames[0]['cy'],frames=frames)
    save(PREP/'transforms.json',transforms);save(PREP/'selected_indices.json',list(range(64,69)));save(PREP/'target_indices.json',list(range(64)))
    caption_src=BASE/'source_64/realworld_white_car_cleanup_64_start19_front_views/captions/realworld_white_car_cleanup_64_start19_front_views/caption_white_car.h5'
    cap=PREP/f'captions/{SCENE}/caption_white_car.h5';cap.parent.mkdir(parents=True);shutil.copyfile(caption_src,cap)
    (PREP/'synthetic_colmap').mkdir(exist_ok=True)
    split=dict(camera_scale=.01,colmap_dir='synthetic_colmap',has_gt=False,image_root='.',initial_object_ply='initial_object/object_local.ply',initialization_mode='stage1_object',metric_scale=1.,opacity_dir=str((rec/'opacity').relative_to(PREP)),prompt_path=str(cap.relative_to(PREP)),render_dir=str((rec/'renders').relative_to(PREP)),scene_id=SCENE,selected_indices_path='selected_indices.json',target_indices_path='target_indices.json',transforms_path='transforms.json')
    save(PREP/'split.json',{'test':{SCENE:split}})
    local_to_world=np.eye(4);local_to_world[:3,:3]*=radius;local_to_world[:3,3]=center
    save(PREP/'normalization.json',dict(center=center.tolist(),radius=radius,local_to_world=local_to_world.tolist(),world_to_local=np.linalg.inv(local_to_world).tolist()))
    g=read(BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json');save(PREP/'up_axis.json',dict(up_axis=g['up']))
    # Verify similarity normalization preserved the previously rendered view.
    orig=np.array(Image.open(OUT/'evaluation/mixed15/orbit/016.png')).astype(float)/255
    normalized=np.array(Image.open(PREP/'images/00016.png')).astype(float)/255
    mae=float(abs(orig-normalized).mean());assert mae<.001,(mae,'Normalization changed rendered appearance')
    save(ROOT/'manifest.json',dict(input_ply=str(original),input_sha256=digest(original),normalized_ply=str(local),normalized_ply_sha256=digest(local),trajectory=str(source),trajectory_sha256=digest(source),camera_count=64,reference_indices=REFS,reference_angles=[trajectory['target_angles_deg'][i] for i in REFS],reference_kind='same PLY renders; no external images',scene=str(PREP),normalization_render_mae=mae,profile='Existing V6-A train-aligned texture-disorder; 4 diffusion steps, seed 0; no 3D distillation',caption_source=str(caption_src),caption_sha256=digest(cap),render_stats=stats,status='prepared'))
    print('PREPARED',PREP,flush=True)
if __name__=='__main__':main()
