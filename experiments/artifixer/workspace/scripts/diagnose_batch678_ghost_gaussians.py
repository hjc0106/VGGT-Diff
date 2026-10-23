"""Frozen-model attribution and temporary removal of endpoint ghost Gaussians.

Color derivatives give exact compositing weights, without enumerating pixel-Gaussian
pairs. Geometry, opacity, depth ordering and all other occluders remain fixed.
"""
from pathlib import Path
import json
import csv
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import vggtdiff_batch678_regularization_ablation as study

ROOT=study.OUT/'outline'
OUT=study.OUT/'ghost_projection_diagnosis_v1'
W,H=832,480
ANCHORS=[22,23]
GROUPS=['upper_outline','middle_outline','lower_outline']


def save(name,value):study.base.save(OUT/name,value)


def regions(mask):
    inner=cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
    outer=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
    return np.stack([inner,~outer,outer&~inner],axis=-1)


def scores(rgb,alpha,gt,mask):
    from vggtdiff_car15_evaluate import metrics
    r=metrics(rgb,alpha,gt,mask)
    for t in [.05,.5]:
        r[f'outside_{t}']=float(((alpha>t)&~mask).sum()/mask.sum())
        r[f'missing_{t}']=float(((alpha<=t)&mask).sum()/mask.sum())
    r['outside_opacity']=float(alpha[~mask].sum()/mask.sum())
    return r


class FrozenRenderer:
    def __init__(self,model):
        self.device='cuda'
        self.n=model.num_gaussians
        self.geometry=dict(means=model.get_positions().detach(),quats=model.get_rotation().detach(),
            scales=model.get_scale().detach(),opacities=model.get_density().flatten().detach())
        self.features=model.get_features().reshape(self.n,-1,3).detach()
        self.degree=int(model.n_active_features)

    def render(self,row,colors=None,remove=None):
        from gsplat import rasterization
        kw=self.geometry.copy()
        if remove is not None:
            kw['opacities']=kw['opacities'].clone();kw['opacities'][remove]=0
        return rasterization(**kw,colors=self.features if colors is None else colors,
            sh_degree=self.degree if colors is None else None,
            viewmats=torch.tensor(row['w2c'],device='cuda',dtype=torch.float32)[None].contiguous(),
            Ks=torch.tensor(row['k'],device='cuda',dtype=torch.float32)[None],width=W,height=H,
            packed=False,near_plane=.001,render_mode='RGB+ED')

    def rgb(self,row,remove=None):
        with torch.no_grad():
            image,alpha,_=self.render(row,remove=remove)
        return image[0,:,:,:3].clamp(0,1).cpu().numpy(),alpha[0,:,:,0].cpu().numpy()

    def attribution(self,row,pixel_weights):
        colors=torch.zeros((self.n,3),device='cuda',requires_grad=True)
        image,alpha,info=self.render(row,colors=colors)
        weights=torch.as_tensor(pixel_weights,device='cuda',dtype=torch.float32)
        objective=(image[0,:,:,:3]*weights).sum()
        grad=torch.autograd.grad(objective,colors)[0]
        expected=(alpha[0]*weights).sum((0,1)).detach()
        error=(grad.sum(0)-expected).abs()
        assert bool((error<.01+1e-4*expected.abs()).all()),(error,expected)
        return grad.detach().cpu().numpy(),alpha[0,:,:,0].detach().cpu().numpy(),float(error.max())

    def tagged(self,row,labels):
        with torch.no_grad():
            image,alpha,_=self.render(row,colors=labels)
        return image[0,:,:,:3].cpu().numpy()


def panel(images,labels,path,size=(416,240)):
    canvas=Image.new('RGB',(size[0]*len(images),size[1]+28),(20,20,20));d=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,labels)):
        if isinstance(im,np.ndarray):im=Image.fromarray(np.uint8(np.clip(im,0,1)*255))
        canvas.paste(im.resize(size),(size[0]*i,28));d.text((size[0]*i+5,7),label,fill='white')
    if path: path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=95)
    return np.asarray(canvas)


def main(cpu=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from threedgrut.model.model import MixtureOfGaussians
    from vggtdiff_selfconsistent_silhouette import render as original_render
    torch.set_num_threads(4)
    study.base.OUT=ROOT;study.base.validate();OUT.mkdir(exist_ok=True)
    result=study.base.read(ROOT/'training/result.json')
    source_hash=study.base.chain.digest(Path(result['checkpoint']))
    if cpu:
        from ghost_cpu_reference import CPUReference
        renderer=CPUReference(result['checkpoint'],OUT)
    else:
        model=MixtureOfGaussians(study.base.load_config())
        model.init_from_checkpoint(torch.load(result['checkpoint'],map_location='cuda',weights_only=False),setup_optimizer=False)
        model.eval();renderer=FrozenRenderer(model)
    rows=study.base.read(ROOT/'inputs.json')['frames']
    masks=[np.asarray(Image.open(r['mask']))>127 for r in rows]
    targets=[np.asarray(Image.open(r['image']).convert('RGB'),dtype=np.float32)/255 for r in rows]
    save('protocol.json',dict(source=str(ROOT),backend='Validated CPU reference' if cpu else 'CUDA gsplat',checkpoint_sha256=source_hash,anchors=ANCHORS,
        anchors_deg=[rows[i]['angle'] for i in ANCHORS],candidate='Smallest score-ranked set explaining >=90% summed anchor outside-alpha mass at alpha>.05',
        attribution='sum_p T_i alpha_i region; CUDA uses diagnostic-color gradients, CPU directly accumulates the same compositing weights',
        boundaries='3x3 eroded interior, dilated exterior, remaining uncertainty band; area metrics use raw masks',
        grouping='Disjoint upper/middle/lower bands from contribution-weighted anchor footprint y, cut at30%/75% target bbox height; not semantic 3D parts',
        removal='Temporary opacity zero for each group and union, no optimizer or saved model mutation',
        visibility='Group-only alpha gives potential visibility; full-scene tag rendering gives actual visibility',
        limits='Generated masks/targets; candidate selection conditioned on two endpoints; no independentGT or causal camera/image diagnosis'))
    original_rgb=[];original_alpha=[];attributions=[];anchor_attribution={};audits=[]
    yy,xx=np.mgrid[:H,:W]
    for j,row in enumerate(rows):
        rgb,alpha=renderer.rgb(row)
        if cpu:
            reference_alpha=np.load(ROOT/'evaluation/analysis_v1/training_alpha_overlays'/f'{j:02d}_alpha.npz')['alpha']
            reference=np.asarray(Image.open(ROOT/'evaluation/training_comparison'/f'{j:03d}.png').crop((832,32,1664,512)),dtype=np.float32)/255
            alpha_error=float(abs(alpha-reference_alpha).mean())
            color_error=float(abs(rgb-reference).mean())
            mask=masks[j]
            area_error=max(abs(float((((alpha>t)&~mask).sum()-((reference_alpha>t)&~mask).sum())/mask.sum())) for t in [.05,.5])
            print('CPU_REFERENCE',j,alpha_error,color_error,area_error,flush=True)
            assert alpha_error<2e-4 and color_error<.001 and area_error<.002,(j,alpha_error,color_error,area_error)
            audits.append(dict(index=j,alpha_mae=alpha_error,rgb_mae_vs_quantized=color_error,outside_area_absolute_error=area_error))
        elif j in [0,22,23]:
            reference,a,_=original_render(model,row)
            err=max(float(abs(rgb-reference).max()),float(abs(alpha-a).max()))
            assert err<2e-5,err
            audits.append(dict(index=j,direct_render_max_error=err))
        mass,al,err=renderer.attribution(row,regions(masks[j]))
        assert np.max(abs(al-alpha))<2e-5
        attributions.append(mass);original_rgb.append(rgb);original_alpha.append(alpha)
        if j in ANCHORS:
            roi=(~cv2.dilate(masks[j].astype('uint8'),np.ones((3,3),np.uint8)).astype(bool))&(alpha>.05)
            weights=np.stack([roi,roi*xx/W,roi*yy/H],axis=-1)
            anchor_attribution[j],_,_=renderer.attribution(row,weights)
        print('ATTRIBUTED',j,'mass_error',err,flush=True)
    # Exact gradient identity also verifies that background/occlusion is accounted for.
    attribution=np.stack(attributions)
    anchor=np.stack([anchor_attribution[i] for i in ANCHORS]).sum(0)
    rank=np.argsort(-anchor[:,0]);cumulative=np.cumsum(anchor[rank,0],dtype=np.float64)
    count=int(np.searchsorted(cumulative,.9*cumulative[-1])+1)
    ids=rank[:count];selected=np.zeros(renderer.n,dtype=bool);selected[ids]=True
    ycentroid=anchor[:,2]/np.maximum(anchor[:,0],1e-12)*H
    y0,y1=np.where(masks[ANCHORS[-1]])[0][[0,-1]]
    cuts=[y0+.30*(y1-y0),y0+.75*(y1-y0)]
    group_index=np.digitize(ycentroid,cuts)
    group_masks=[selected&(group_index==i) for i in range(3)]
    assert sum(int(m.sum()) for m in group_masks)==len(ids)
    variants={name:torch.as_tensor(m,device=renderer.device) for name,m in zip(GROUPS,group_masks)}
    variants['all_selected']=torch.as_tensor(selected,device=renderer.device)
    labels=torch.tensor(np.stack(group_masks,-1),device=renderer.device,dtype=torch.float32)
    np.savez_compressed(OUT/'gaussian_attribution.npz',region_mass=attribution,anchor_mass=anchor,
        selected_ids=ids,group_index=np.where(selected,group_index,-1),positions=renderer.geometry['means'].cpu().numpy(),
        scales=renderer.geometry['scales'].cpu().numpy(),opacities=renderer.geometry['opacities'].cpu().numpy())
    save('selection.json',dict(total_gaussians=renderer.n,selected=len(ids),score_coverage=float(anchor[ids,0].sum()/anchor[:,0].sum()),
        anchor_mass_total=float(anchor[:,0].sum()),groups={g:int(m.sum()) for g,m in zip(GROUPS,group_masks)},
        y_band_cuts=cuts,gaussian_id_order='Checkpoint tensor order, zero-based; do not apply IDs to other models'))
    fit=[];group_stats=[];display=[];union_rgb=[];heat_out=np.zeros((3,24));heat_vis=np.zeros((3,24));heat_inside=np.zeros((3,24))
    for j,row in enumerate(rows):
        mask=masks[j];gt=targets[j];rgb=original_rgb[j];alpha=original_alpha[j]
        fit.append(dict(index=j,angle=row['angle'],variant='original',**scores(rgb,alpha,gt,mask)))
        variants_rgb=[]
        for name,remove in variants.items():
            r,a=renderer.rgb(row,remove)
            fit.append(dict(index=j,angle=row['angle'],variant=name,**scores(r,a,gt,mask)))
            variants_rgb.append(r)
            if name=='all_selected':union_rgb.append(r)
        tag=renderer.tagged(row,labels)
        # Marker RGB channels retain every occluding Gaussian, including unselected ones.
        colored=np.stack([tag[:,:,0]+tag[:,:,1],tag[:,:,1]+tag[:,:,2],tag[:,:,0]+tag[:,:,2]],axis=-1)
        overlay=np.clip(.5*rgb+colored,0,1)
        display.append(panel([gt*mask[:,:,None],rgb,overlay,union_rgb[-1]],
            [f'Target | {row["angle"]:.2f}deg','Original outline model','Selected: upper magenta / middle yellow / lower cyan','Remove union (diagnostic)'],OUT/'training_panels'/f'{j:02d}.jpg'))
        panel([rgb]+variants_rgb,['Original']+[f'Remove {k}' for k in variants],OUT/'group_removals'/f'{j:02d}.jpg')
        for gi,(g,m) in enumerate(zip(GROUPS,group_masks)):
            masses=attribution[j,m].sum(0)
            actual=float(tag[:,:,gi].sum())
            assert abs(float(masses.sum())-actual)<.03+actual*1e-4
            # Render this group alone only to measure the effect of occlusion.
            with torch.no_grad():
                _,isolated,_=renderer.render(row,colors=torch.ones_like(labels),remove=~variants[g])
            potential=float(isolated.sum())
            visible_fraction=actual/max(potential,1e-12)
            stat=dict(index=j,angle=row['angle'],group=g,inside_mass=float(masses[0]),outside_mass=float(masses[1]),
                boundary_mass=float(masses[2]),visible_mass=actual,isolated_mass=potential,visible_to_isolated=visible_fraction,
                outside_fraction=float(masses[1]/max(actual,1e-12)),inside_fraction=float(masses[0]/max(actual,1e-12)),
                visible_mass_over_target_area=actual/float(mask.sum()))
            group_stats.append(stat);heat_out[gi,j]=stat['outside_fraction'];heat_vis[gi,j]=visible_fraction;heat_inside[gi,j]=stat['inside_fraction']
        print('REMOVAL_CHECKED',j,flush=True)
    save('per_view_metrics.json',fit);save('group_view_contributions.json',group_stats)
    for filename,records in [('per_view_metrics.csv',fit),('group_view_contributions.csv',group_stats)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    fig,axs=plt.subplots(3,1,figsize=(15,6.5),layout='constrained')
    for ax,data,title in zip(axs,[heat_out,heat_inside,heat_vis],['Visible contribution outside dilated target mask','Visible contribution inside eroded target mask','Visibility: full-scene contribution / group-only contribution']):
        im=ax.imshow(data,vmin=0,vmax=1,aspect='auto',cmap='magma');ax.set_yticks(range(3),GROUPS);ax.set_xticks(range(24),[f'{r["angle"]:.1f}' for r in rows],rotation=60);ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.savefig(OUT/'conflict_heatmap.png',dpi=160);plt.close(fig)
    # Six fixed examples and all24 contacts, with identical image scaling.
    for name,indices in [('examples',[0,7,8,15,22,23]),('all24',list(range(24)))]:
        canv=Image.new('RGB',(display[0].shape[1],display[0].shape[0]*len(indices)))
        for ri,j in enumerate(indices):canv.paste(Image.fromarray(display[j]),(0,display[j].shape[0]*ri))
        canv.save(OUT/f'{name}.jpg',quality=95)
    summary={}
    for variant in ['original']+list(variants):
        subset=[r for r in fit if r['variant']==variant]
        summary[variant]={k:float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in ['index','angle','variant']}
        summary[variant]['endpoint']=subset[-1]
        if variant!='original':
            changes=[r['foreground_psnr']-b['foreground_psnr'] for r,b in zip(subset,[r for r in fit if r['variant']=='original'])]
            summary[variant]['worst_psnr_delta']=float(min(changes));summary[variant]['worst_psnr_index']=int(np.argmin(changes))
    save('summary.json',summary)
    print('SUMMARY',summary,flush=True)
    videos={}
    sets=study.base.read(ROOT/'evaluation/cameras.json')
    for name,cameras in sets.items():
        path=OUT/f'{name}_removal_comparison.mp4'
        with imageio.get_writer(str(path),fps=12,macro_block_size=16) as writer:
            for i,row in enumerate(cameras):
                before,_=renderer.rgb(row);after,_=renderer.rgb(row,variants['all_selected'])
                frame=panel([before,after],[f'Original | {row["angle"]:.2f}deg','Remove selected union (diagnostic)'],None,size=(832,480))
                # Pad explicitly to preserve pixels and avoid automatic video scaling.
                padded=np.zeros((512,1664,3),dtype=np.uint8);padded[:508]=frame;writer.append_data(padded)
        cap=cv2.VideoCapture(str(path));count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==len(cameras)
        videos[name]=dict(frames=count,sha256=study.base.chain.digest(path));print('VIDEO',name,count,flush=True)
    study.base.validate();assert study.base.chain.digest(Path(result['checkpoint']))==source_hash
    save('audit.json',dict(state='passed',direct_render_checks=audits,attribution_mass_identity='Passed all24 views and two anchors',
        tagged_group_vs_per_gaussian_sum='Passed all24 x3 groups',videos=videos,
        checkpoint_unchanged=True,source_model_ply_unchanged=True,no_optimizer_steps=True))
    save('status.json',dict(state='completed_pending_review',summary=str(OUT/'summary.json')))
    print('DONE',OUT,flush=True)


if __name__=='__main__':
    # Install repository backend/import path without changing any shared files.
    study.ORIGINAL_TRAINER()
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');a=p.parse_args()
    main(cpu=a.cpu)
