"""Exploratory native-coordinate residual plots; no image alignment or refitting."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import vggtdiff_matched_gt as job
from vggtdiff_white_car import save
import crossview_geometry_eval as geom


def run(seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    job.validate()
    targets=job.read(job.OUT/'targets.json')['frames']
    out=job.OUT/'spatial_diagnostics';out.mkdir(exist_ok=True)
    save(out/'scope.json',dict(scope='Exploratory descriptive diagnostic after first-seed results; not a preregistered primary endpoint.',
        method='Native GT/generated coordinates only; no translation, scale, homography, or camera refit applied.'))
    rows=[]
    for seed in seeds:
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        for i,(r,ax) in enumerate(zip(targets,axes.flat)):
            with np.load(job.OUT/f'evaluation/seed_{seed}'/f"same_view_{r['original_frame']}.npz") as d:
                a,b=d['gt_xy'],d['generated_xy']
            delta=b-a
            gt=geom.read_mask(r['mask'])>0
            generated=geom.read_mask(job.OUT/f'seed_{seed}/masks'/f'{i:02d}.png')>0
            ys,xs=np.where(gt);yg,xg=np.where(generated)
            shift=np.median(delta,axis=0) if len(delta) else [None,None]
            rows.append(dict(seed=seed,frame=r['original_frame'],n_matches=len(delta),
                median_dx_px=float(shift[0]) if shift[0] is not None else None,
                median_dy_px=float(shift[1]) if shift[1] is not None else None,
                generated_to_gt_mask_area_ratio=float(generated.sum()/gt.sum()),
                mask_centroid_dx_px=float(xg.mean()-xs.mean()),mask_centroid_dy_px=float(yg.mean()-ys.mean())))
            ax.imshow(cv2.cvtColor(geom.read_image(r['image']),cv2.COLOR_BGR2RGB))
            ax.contour(gt,levels=[.5],colors=['cyan'],linewidths=.7)
            ax.contour(generated,levels=[.5],colors=['red'],linewidths=.7)
            if len(delta):ax.quiver(a[:,0],a[:,1],delta[:,0],delta[:,1],angles='xy',scale_units='xy',scale=1,color='yellow',width=.004)
            ax.set_xlim(max(0,xs.min()-30),min(832,xs.max()+30));ax.set_ylim(min(480,ys.max()+30),max(0,ys.min()-30))
            ax.set_title(f"GT frame {r['original_frame']} | n={len(delta)}",fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'Seed {seed}: cyan GT mask, red generated mask, yellow raw correspondence displacement (no alignment)',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/f'seed_{seed}.png',dpi=140);plt.close(fig)
    geom.write_csv(out/'spatial_bias.csv',rows,list(rows[0]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=job.SEEDS)
    run(parser.parse_args().seeds)
