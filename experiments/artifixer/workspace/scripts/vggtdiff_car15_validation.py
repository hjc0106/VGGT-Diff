"""Independent real frames for appearance validation; localized via real 3D-2D matches."""
import sys,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
from PIL import Image
from vggtdiff_car15_prepare import OUT,BASE,REPO,PARENT,read
from vggtdiff_white_car import save,digest
import vggtdiff_realvideo_unseen as e
sys.path.insert(0,str(REPO/'worldact'))


def main():
    root=OUT/'validation';(root/'full').mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True);(root/'masks').mkdir(exist_ok=True)
    video=REPO/'pipeline_exp/realworld/scene-1-nofull/scene-1-wcar.mp4'
    for frame in [600,1080]:
        dest=root/'full'/f'{frame}.png'
        if not dest.exists():subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(video),'-vf',f'select=eq(n\\,{frame})','-frames:v','1',str(dest)],check=True)
    meta=read(PARENT/'camera_estimate.json');g=read(PARENT/'geometry_check.json');t=read(PARENT/'trajectory.json')
    sift=cv2.SIFT_create(nfeatures=10000);cv2.setNumThreads(4);sources=[]
    for i,item in enumerate(meta['sources']):
        im=cv2.imread(str(e.INPUT/item['full_image']));im=cv2.resize(im,(1280,720));kp,des=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None);sources.append((kp,des))
    results=[]
    with np.load(BASE/'vggtdiff_realvideo_unseen_right_v1/omega_dense.npz') as dense:
        kfull=np.median(dense['k_original'],axis=0);ksmall=kfull.copy();ksmall[:2]/=3
        for frame in [600,1080]:
            full=cv2.imread(str(root/'full'/f'{frame}.png'));small=cv2.resize(full,(1280,720));kp,des=sift.detectAndCompute(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),None);points=[];pixels=[]
            for i,(kr,dr) in enumerate(sources):
                matches=cv2.BFMatcher().knnMatch(dr,des,k=2)
                good=[m for m,n in matches if m.distance<.7*n.distance]
                inv=np.linalg.inv(dense['processed_to_original'][i]);conf=dense['confidence'][i];threshold=np.quantile(conf,.6)
                for m in good:
                    xy=np.array([*kr[m.queryIdx].pt,1.]);xy[:2]*=3;p=inv@xy;u,v=np.rint(p[:2]/p[2]).astype(int)
                    if 0<=u<688 and 0<=v<384 and conf[v,u]>=threshold:
                        xyz=dense['world'][i,v,u]
                        if np.isfinite(xyz).all():points.append(xyz);pixels.append(kp[m.trainIdx].pt)
            xyz=np.array(points,dtype=np.float32);uv=np.array(pixels,dtype=np.float32)
            ok,rv,tv,inliers=cv2.solvePnPRansac(xyz,uv,ksmall,None,iterationsCount=10000,reprojectionError=2.5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            print('PNP_CANDIDATE',frame,len(inliers) if ok else 0,flush=True)
            assert ok and len(inliers)>=12,('Pose localization failed',frame,len(inliers) if ok else 0)
            ids=inliers[:,0];rv,tv=cv2.solvePnPRefineLM(xyz[ids],uv[ids],ksmall,None,rv,tv)
            errors=np.linalg.norm(cv2.projectPoints(xyz[ids],rv,tv,ksmall,None)[0][:,0]-uv[ids],axis=1)
            print('PNP_ERRORS',float(np.median(errors)),float(np.quantile(errors,.95)), 'span',np.ptp(uv[ids],axis=0).tolist(),flush=True)
            assert np.median(errors)<2
            w=np.eye(4);w[:3,:3]=cv2.Rodrigues(rv)[0];w[:3,3]=tv[:,0];pos=np.linalg.inv(w)[:3,3]
            neww=e.look_at(pos,np.array(g['center']),np.array(g['up']));newk=np.median(np.array(t['source_intrinsics']),axis=0);newk[0,2]=415.5;newk[1,2]=239.5
            h=newk@neww[:3,:3]@np.linalg.inv(w[:3,:3])@np.linalg.inv(kfull)
            warped=cv2.warpPerspective(full,h,(832,480));valid=cv2.warpPerspective(np.ones(full.shape[:2],np.uint8),h,(832,480),flags=cv2.INTER_NEAREST)
            assert valid.mean()>.999
            path=root/'images'/f'{frame}.png';Image.fromarray(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)).save(path)
            results.append(dict(original_frame=frame,name=path.name,path=str(path),sha256=digest(path),w2c=neww.tolist(),k=newk.tolist(),full_w2c=w.tolist(),full_k=kfull.tolist(),homography=h.tolist(),matches=len(xyz),inliers=len(ids),median_reprojection_error_1280_px=float(np.median(errors)),p95_reprojection_error_1280_px=float(np.quantile(errors,.95)),role='Validation only; not in six conditioning references or fifteen training images'))
            print('LOCALIZED',frame,len(ids),float(np.median(errors)),flush=True)
    save(root/'cameras.json',dict(frames=results,calibration_caution='PnP against estimated Omega real-source geometry; pose/depth errors remain; evaluates real-side appearance, not true hidden-side geometry'))
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    segmenter=SAM3Segmenter(SimpleNamespace(source_dir=str(REPO/'thirdparty/sam3'),checkpoint_path=str(REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
    for r in results:
        rgb=np.array(Image.open(r['path']).convert('RGB'));path=root/'masks'/r['name']
        if not path.exists():
            with torch.inference_mode():cs=segmenter.segment_candidates(torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255,'white car')
            choices=[]
            for c in ([] if cs is None else cs.candidates):
                m=c.mask.cpu().numpy()>.5
                if .08<m.mean()<.7 and m[140:].sum()/max(1,m.sum())>.8:choices.append((m.sum(),m))
            assert choices;mask=max(choices,key=lambda p:p[0])[1];Image.fromarray(mask.astype('uint8')*255).save(path)
        mask=np.array(Image.open(path))>127;overlay=rgb.copy();overlay[mask]=(rgb[mask]*.6+np.array([0,255,0])*.4).astype('uint8');Image.fromarray(overlay).save(root/f'overlay_{r["original_frame"]}.png')
    e.contact([root/f'overlay_{f}.png' for f in [600,1080]],['Independent real frame 600','Independent real frame 1080'],root/'validation_contact.jpg',columns=2)
    print('VALIDATION_PREPARED',flush=True)


if __name__=='__main__':main()
