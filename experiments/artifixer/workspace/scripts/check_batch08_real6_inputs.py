"""Vehicle-only, unfiltered fixed-camera diagnostic on six real source views."""
from pathlib import Path
from types import SimpleNamespace
import itertools
import cv2
import numpy as np
from PIL import Image
import analyze_batch08_real_anchor_results as evaluator
import vggtdiff_batch08_real6_continuous as job

OUT = job.OUT / 'input_geometry_check'


def prepare():
    import torch
    from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    manifest, trajectory = job.chain.validate(job.OUT)
    OUT.mkdir(exist_ok=True)
    frames, overlays, labels = [], [], []
    segmenter = None
    for i, source in enumerate(manifest['sources']):
        image = job.OUT/'source_views'/f'{i:02d}.png'
        mask_path = OUT/f'{i:02d}_mask.png'
        rgb = np.asarray(Image.open(image).convert('RGB'))
        if not mask_path.exists():
            if i >= 2:
                job.chain.frozen_copy(job.mixed.OUT/'input_geometry_check'/f'{i:02d}_mask.png',mask_path)
            else:
                if segmenter is None:
                    segmenter = SAM3Segmenter(SimpleNamespace(source_dir=str(job.chain.REPO/'thirdparty/sam3'),
                        checkpoint_path=str(job.chain.REPO/'thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt'),
                        device='cuda',confidence_threshold=.15,instance_strategy='best_score'))
                mask = evaluator.choose_vehicle(segmenter,rgb)
                Image.fromarray(mask.astype('uint8')*255).save(mask_path)
        mask = evaluator.geom.read_mask(mask_path)>0
        ys,xs = np.where(mask)
        assert len(xs)>1000 and xs.min()>5 and xs.max()<826 and ys.min()>5 and ys.max()<474
        frames.append(dict(index=i,original_frame=i,image=str(image),mask=str(mask_path),
            K=source['intrinsics'],T=source['w2c'],azimuth_deg=source['angle'],video_frame=source['original_frame'],
            sha256=job.chain.digest(image),mask_sha256=job.chain.digest(mask_path)))
        over=rgb.copy();over[mask]=(.55*rgb[mask]+.45*np.array([0,255,0])).astype('uint8')
        dest=OUT/f'{i:02d}_overlay.jpg';Image.fromarray(over).save(dest)
        overlays.append(dest);labels.append(f"Real {source['original_frame']} | {source['angle']:.3f} deg")
        print('INPUT_MASK',i,source['original_frame'],len(xs),flush=True)
    del segmenter
    torch.cuda.empty_cache()
    job.chain.montage(overlays,labels,OUT/'mask_review.jpg',3,(832,480))
    job.chain.save(OUT/'frames.json',dict(frames=frames))
    files=[Path(__file__).resolve(),Path(evaluator.__file__).resolve(),job.OUT/'manifest.json',job.OUT/'trajectory.json',OUT/'frames.json']
    files += [Path(f[k]) for f in frames for k in ['image','mask']]
    job.chain.save(OUT/'manifest.json',dict(files=[dict(path=str(p),sha256=job.chain.digest(p)) for p in files]))


def evaluate():
    for row in job.chain.read(OUT/'manifest.json')['files']:
        assert job.chain.digest(Path(row['path']))==row['sha256']
    assert job.chain.read(OUT/'mask_review.json')['state']=='accepted'
    cv2.setNumThreads(4)
    frames=[evaluator.matched.feature_frame(r,OUT/'features') for r in job.chain.read(OUT/'frames.json')['frames']]
    matches=evaluator.matched.build_matches(frames,OUT)
    rows=[]
    for a,b in itertools.combinations(frames,2):
        ia,ib=matches[(a['original_frame'],b['original_frame'])]
        r=evaluator.geom.pair_record(a['index'],b['index'],a['xy'][ia],b['xy'][ib],a['mask'],b['mask'],a['K'],a['T'],b['K'],b['T'])
        r['azimuth_gap_deg']=evaluator.geom.angular_gap(a['azimuth_deg'],b['azimuth_deg'])
        rows.append(r)
    evaluator.geom.write_csv(OUT/'pairs.csv',rows,evaluator.geom.PAIR_FIELDS)
    evaluator.geom._write_overlays(OUT,frames,matches,rows)
    summary=dict(scope='Six conditioning images only, not independent generated-image GT',
        protocol='Vehicle-only SIFT12000/BGR grayscale, mutual ratio.75, no epipolar/RANSAC filtering, no camera fitting;30-match support rule',
        primary15=evaluator.dense.pair_summary([r for r in rows if r['azimuth_gap_deg']<=15]),
        all=evaluator.dense.pair_summary(rows),camera_status='Estimated registered SfM + common background Sim3; not pose GT')
    job.chain.save(OUT/'summary.json',summary)
    print('INPUT_SUMMARY',summary,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','evaluate'])
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else evaluate()
