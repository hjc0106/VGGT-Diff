"""Prepare masked real6/generated9 images, fixed cameras and shared geometry."""
import json,sys,shutil,argparse,struct
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import cv2

from vggtdiff_white_car import BASE,REPO,digest,save
import vggtdiff_realvideo_unseen as e
sys.path.insert(0,str(REPO/'worldact'))
PARENT=BASE/'vggtdiff_realvideo_unseen_right_near_v2'
SHORT=BASE/'vggtdiff_realvideo_shared_context9_eight_v1'
OUT=BASE/'vggtdiff_car15_reconstruction_v1'


def read(p):return json.loads(p.read_text())


def inputs():
    OUT.mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True);(OUT/'masks').mkdir(exist_ok=True)
    a=read(PARENT/'trajectory.json');b=read(SHORT/'trajectory.json');rows=[]
    for i in range(15):
        real=i<6;j=i if real else i-6
        src=PARENT/'source_views'/f'{j:02d}.png' if real else SHORT/'raw'/f'{j:02d}.png'
        dst=OUT/'images'/f'{i:02d}.png'
        if not dst.exists():shutil.copyfile(src,dst)
        assert digest(src)==digest(dst)
        w=a['source_w2c'][j] if real else b['target_w2c'][j];k=a['source_intrinsics'][j] if real else b['target_intrinsics'][j]
        rows.append(dict(index=i,name=f'{i:02d}.png',kind='real' if real else 'generated',source=str(src),sha256=digest(src),w2c=w,k=k,angle=None if real else b['target_angles_deg'][j]))
    m=dict(frames=rows,width=832,height=480,coordinates='Original VGGT-Omega world; OpenCV W2C; no crop or resize',generated_source=str(SHORT),frozen_files=[dict(path=str(Path(x['source'])),sha256=x['sha256']) for x in rows])
    if (OUT/'inputs.json').exists():assert read(OUT/'inputs.json')==m
    else:save(OUT/'inputs.json',m)
    return rows


def masks():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    rows=inputs();seg=SAM3Segmenter(SimpleNamespace(source_dir=str(REPO/'thirdparty/sam3'),checkpoint_path=str(REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
    cv2.setNumThreads(4);torch.set_num_threads(4);stats=[];tiles=[]
    for r in rows:
        i=r['index'];im=np.array(Image.open(OUT/'images'/r['name']).convert('RGB'));p=OUT/'masks'/r['name']
        if not p.exists():
            with torch.inference_mode():candidates=seg.segment_candidates(torch.from_numpy(im.copy()).permute(2,0,1).float().cuda()/255,'white car')
            choices=[]
            for c in ([] if candidates is None else candidates.candidates):
                m=c.mask.cpu().numpy()>.5;area=m.mean();ys,xs=np.where(m)
                if not len(xs):continue
                below=m[140:].sum()/max(1,m.sum());center=np.array([xs.mean(),ys.mean()]);score=float(c.score)+2*area+below-.0008*np.linalg.norm(center-[416,330])
                if .05<area<.7 and below>.8:choices.append((score,m))
            if not choices:raise RuntimeError(f'No eligible foreground white car: {i}')
            m=max(choices,key=lambda x:x[0])[1]
            Image.fromarray(m.astype('uint8')*255).save(p)
        mask=np.array(Image.open(p).convert('L'))>127;ys,xs=np.where(mask);assert len(xs)>1000
        overlay=im.copy();overlay[mask]=(im[mask]*.6+np.array([0,255,0])*.4).astype('uint8')
        cv2.drawContours(overlay,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,0,0),1)
        tile=Image.new('RGB',(416,272),'#181818');tile.paste(Image.fromarray(overlay).resize((416,240)),(0,32));ImageDraw.Draw(tile).text((8,9),f'{i:02d} {r["kind"]} | area {mask.mean():.3f}',fill='white');tiles.append(tile)
        rgba=np.dstack([im,mask.astype('uint8')*255]);(OUT/'rgba').mkdir(exist_ok=True);Image.fromarray(rgba).save(OUT/'rgba'/r['name'])
        stats.append(dict(index=i,area=float(mask.mean()),bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],sha256=digest(p)))
        print('MASK',i,stats[-1],flush=True)
    sheet=Image.new('RGB',(416*3,272*5))
    for i,t in enumerate(tiles):sheet.paste(t,(i%3*416,i//3*272))
    sheet.save(OUT/'masks_contact.jpg',quality=95);save(OUT/'masks.json',dict(frames=stats,visual_review='pending'))


def write_cloud(path,xyz,rgb):
    from plyfile import PlyData,PlyElement
    a=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for j,k in enumerate(['x','y','z']):a[k]=xyz[:,j]
    for j,k in enumerate(['red','green','blue']):a[k]=rgb[:,j]
    PlyData([PlyElement.describe(a,'vertex')]).write(path)


def initialize():
    rows=inputs();g=read(PARENT/'geometry_check.json');parts=[];colors=[]
    dense=BASE/'vggtdiff_realvideo_unseen_right_v1/omega_dense.npz'
    with np.load(dense) as d:
        for i,r in enumerate(rows[:6]):
            xyz=d['world'][i].reshape(-1,3);conf=d['confidence'][i].reshape(-1)
            uv,z=e.project(xyz,np.array(r['w2c']),np.array(r['k']));valid=np.isfinite(xyz).all(1)&np.isfinite(uv).all(1)&(z>0)
            pixels=np.rint(np.nan_to_num(uv)).astype(int);valid&=(pixels[:,0]>=0)&(pixels[:,0]<832)&(pixels[:,1]>=0)&(pixels[:,1]<480)
            ids=np.flatnonzero(valid);mask=np.array(Image.open(OUT/'masks'/r['name']))>127
            ids=ids[mask[pixels[ids,1],pixels[ids,0]]];ids=ids[conf[ids]>=np.quantile(conf[ids],.4)]
            rgb=np.array(Image.open(OUT/'images'/r['name']).convert('RGB'));parts.append(xyz[ids]);colors.append(rgb[pixels[ids,1],pixels[ids,0]])
    xyz=np.concatenate(parts);rgb=np.concatenate(colors);center=np.array(g['center']);axes=np.array([g['front'],g['left'],g['up']]).T
    local=(xyz-center)@axes;lo,hi=np.quantile(local,[.01,.99],axis=0);extent=hi-lo
    keep=((local>lo-.15*extent)&(local<hi+.15*extent)).all(1);xyz=xyz[keep];rgb=rgb[keep]
    vox=np.floor(xyz/(max(extent)/180)).astype(int);_,ids=np.unique(vox,axis=0,return_index=True);xyz=xyz[ids];rgb=rgb[ids]
    distance=cKDTree(xyz).query(xyz,k=8)[0][:,-1];keep=distance<np.quantile(distance,.98);xyz=xyz[keep];rgb=rgb[keep]
    rng=np.random.default_rng(20261021)
    if len(xyz)>25000:ids=rng.choice(len(xyz),25000,replace=False);xyz=xyz[ids];rgb=rgb[ids]
    n=len(xyz);seeds=rng.uniform(lo-.1*extent,hi+.1*extent,(4000,3))@axes.T+center
    seedcolors=rgb[cKDTree(xyz).query(seeds)[1]];xyz=np.r_[xyz,seeds];rgb=np.r_[rgb,seedcolors]
    assert n>1000 and np.isfinite(xyz).all();write_cloud(OUT/'initial_points.ply',xyz,rgb)
    save(OUT/'initialization.json',dict(real_surface_points=n,volume_seed_points=4000,seed_opacity=.01,seed_prior='Uniform low-opacity volume seeds inside expanded observed-car OBB, not measured hidden geometry',cloud=str(OUT/'initial_points.ply'),sha256=digest(OUT/'initial_points.ply'),bounds_world=[xyz.min(0).tolist(),xyz.max(0).tolist()],dense_source=str(dense)))
    for name,subset in [('real6',rows[:6]),('mixed15',rows)]:
        root=OUT/'datasets'/name;(root/'images').mkdir(parents=True,exist_ok=True);sparse=root/'sparse/0';sparse.mkdir(parents=True,exist_ok=True)
        cams=[];images=[]
        for j,r in enumerate(subset):
            k=np.array(r['k']);w=np.array(r['w2c']);q=Rotation.from_matrix(w[:3,:3]).as_quat();q=np.r_[q[3],q[:3]];cid=j+1
            cams.append(f'{cid} OPENCV 832 480 {k[0,0]:.17g} {k[1,1]:.17g} {k[0,2]:.17g} {k[1,2]:.17g} 0 0 0 0')
            images.append(' '.join(map(str,[cid,*q,*w[:3,3],cid,r['name']]))+'\n')
            shutil.copyfile(OUT/'images'/r['name'],root/'images'/r['name']);shutil.copyfile(OUT/'masks'/r['name'],root/'images'/f'{Path(r["name"]).stem}_mask.png')
        (sparse/'cameras.txt').write_text('\n'.join(cams)+'\n');(sparse/'images.txt').write_text('\n'.join(images)+'\n')
        with (sparse/'cameras.bin').open('wb') as f:
            f.write(struct.pack('<Q',len(subset)))
            for j,r in enumerate(subset):
                k=np.array(r['k']);params=[k[0,0],k[1,1],k[0,2],k[1,2],0,0,0,0]
                f.write(struct.pack('<iiQQ',j+1,4,832,480));f.write(struct.pack('<8d',*params))
        with (sparse/'images.bin').open('wb') as f:
            f.write(struct.pack('<Q',len(subset)))
            for j,r in enumerate(subset):
                w=np.array(r['w2c']);q=Rotation.from_matrix(w[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
                f.write(struct.pack('<i7di',j+1,*q,*w[:3,3],j+1));f.write(r['name'].encode()+b'\0');f.write(struct.pack('<Q',0))
        (sparse/'points3D.txt').write_text('# Initialization uses explicit fused point cloud\n')
        shutil.copyfile(OUT/'initial_points.ply',root/'point_cloud.ply')
    save(OUT/'preparation.json',dict(state='prepared',images=15,masks=15,surface_points=n,seed_points=4000,init_shared=True,per_image_intrinsics=True))
    print('PREPARED',n,OUT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['masks','initialize']);args=p.parse_args();globals()[args.action]()
