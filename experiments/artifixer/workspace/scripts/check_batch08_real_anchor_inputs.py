"""Diagnostic vehicle mask/epipolar check on prepared six source views only."""
from pathlib import Path
from types import SimpleNamespace
import itertools
import cv2
import numpy as np
from PIL import Image
import crossview_geometry_eval as geom
import crossview_geometry_dense_gt as dense
import vggtdiff_batch08_real_anchors as job


def main():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    from vggtdiff_car15_groupa_prepare import choose_mask
    torch.set_num_threads(4);cv2.setNumThreads(4)
    m,t=job.chain.validate(job.OUT);root=job.OUT/'input_geometry_check';root.mkdir(exist_ok=True)
    segmenter=None;frames=[];overlays=[]
    for i,s in enumerate(m['sources']):
        image=job.OUT/'source_views'/f'{i:02d}.png';mask_path=root/f'{i:02d}_mask.png'
        rgb=np.asarray(Image.open(image).convert('RGB'))
        if not mask_path.exists():
            if s['role']=='generated':
                cached=job.chain.BASE/'crossview_geometry_batch06_batch07_continuous_v1/masks'/f"{8+s['source_local_index']:02d}.png"
                job.chain.frozen_copy(cached,mask_path)
            else:
                if segmenter is None:
                    segmenter=SAM3Segmenter(SimpleNamespace(source_dir=str(job.chain.REPO/'thirdparty/sam3'),checkpoint_path=str(job.chain.REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
                mask=choose_mask(segmenter,rgb);Image.fromarray(mask.astype('uint8')*255).save(mask_path)
        mask=geom.read_mask(mask_path);ys,xs=np.where(mask>0)
        assert len(xs)>1000 and xs.min()>5 and xs.max()<826 and ys.min()>5 and ys.max()<474,'Vehicle reference may be cut off'
        xy,des=geom.extract_features(rgb,mask)
        f=dict(original_frame=i,role=s['role'],image=str(image),mask=mask,K=np.array(s['intrinsics']),T=np.array(s['w2c']),azimuth_deg=s['angle'],xy=xy,descriptors=des)
        frames.append(f)
        over=rgb.copy();inside=mask>0;over[inside]=(.55*rgb[inside]+.45*np.array([0,255,0])).astype('uint8')
        p=root/f'{i:02d}_overlay.jpg';Image.fromarray(over).save(p);overlays.append(p)
        print('CHECK_SOURCE',i,s['role'],'mask area',len(xs),'features',len(xy),flush=True)
    del segmenter;torch.cuda.empty_cache()
    job.chain.montage(overlays,[f"{i} | {s['role']} | {s['angle']:.2f}deg" for i,s in enumerate(m['sources'])],root/'mask_review.jpg',3,(832,480))
    rows=[]
    for a,b in itertools.combinations(frames,2):
        ia,ib=geom.mutual_ratio_matches(a['descriptors'],b['descriptors'])
        r=geom.pair_record(a['original_frame'],b['original_frame'],a['xy'][ia],b['xy'][ib],a['mask'],b['mask'],a['K'],a['T'],b['K'],b['T'])
        r['azimuth_gap_deg']=geom.angular_gap(a['azimuth_deg'],b['azimuth_deg']);r['group']='mixed' if a['role']!=b['role'] else a['role']
        rows.append({k:v for k,v in r.items() if k!='errors'})
    geom.write_csv(root/'pairs.csv',rows,geom.PAIR_FIELDS+['group'])
    summary={g:dense.pair_summary([r for r in rows if r['group']==g and r['azimuth_gap_deg']<=15]) for g in ['real','generated','mixed']}
    job.chain.save(root/'summary.json',dict(scope='Six prepared references only, not new generated targets; descriptive support, no quality acceptance thresholds.',groups15=summary,pairs=rows))
    print('INPUT_GEOMETRY',summary,flush=True)

if __name__=='__main__':main()
