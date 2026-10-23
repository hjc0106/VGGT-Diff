"""Read-only render diagnostics for the completed24-view reconstruction."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import vggtdiff_batch678_reconstruct as exp

OUT=exp.OUT/'evaluation/analysis_v1'


def outline_diagnostics(alpha,mask,settings=None):
    settings=settings or dict(outline_foreground_normalized=False,outline_weight=.2)
    expanded=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
    integral=float(alpha[~expanded].sum())
    background_loss=integral/max(int((~expanded).sum()),1)
    foreground_normalized=settings['outline_foreground_normalized']
    loss=integral/max(int(mask.sum() if foreground_normalized else (~expanded).sum()),1)
    return dict(training_outside_opacity_loss=loss,
        training_outside_opacity_contribution=settings['outline_weight']*loss,
        training_outside_opacity_background_normalized_reference=background_loss,
        training_outside_opacity_denominator='foreground_mask_area' if foreground_normalized else 'expanded_background_area',
        training_outside_opacity_weight=settings['outline_weight'])


def main():
    import torch
    from plyfile import PlyData
    import vggtdiff_car15_train
    from threedgrut.model.model import MixtureOfGaussians
    from vggtdiff_selfconsistent_silhouette import render
    exp.validate();torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    result=exp.read(exp.OUT/'training/result.json')
    settings=exp.read(exp.OUT/'loss_settings.json') if (exp.OUT/'loss_settings.json').exists() else None
    assert exp.chain.digest(exp.OUT/'model.ply')==result['ply_sha256']
    model=MixtureOfGaussians(exp.load_config())
    model.init_from_checkpoint(torch.load(result['checkpoint'],map_location='cuda',weights_only=False),setup_optimizer=False)
    model.eval();rows=exp.read(exp.OUT/'inputs.json')['frames'];scores=[];overlays=[]
    for i,r in enumerate(rows):
        rgb,alpha,depth=render(model,r)
        assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth).all()
        mask=np.asarray(Image.open(r['mask']))>127;area=int(mask.sum())
        score=dict(index=i,batch=r['batch'],angle=r['angle'],mask_area=area,
            outside_opacity_over_mask_area=float(alpha[~mask].sum()/area),
            outside_loss_background_normalized=float(alpha[~mask].sum()/(~mask).sum()),
            background_to_foreground_area_ratio=float((~mask).sum()/area))
        for threshold in [.05,.5]:
            pred=alpha>threshold
            score[f'outside_area_over_mask_{threshold}']=float((pred&~mask).sum()/area)
            score[f'missing_mask_fraction_{threshold}']=float((mask&~pred).sum()/area)
            score[f'iou_{threshold}']=float((pred&mask).sum()/(pred|mask).sum())
        # Match each branch's actual denominator and coefficient.
        score.update(outline_diagnostics(alpha,mask,settings))
        pred=alpha>.5;over=np.uint8(np.clip(rgb,0,1)*255)
        over[pred&~mask]=np.uint8(.35*over[pred&~mask]+.65*np.array([255,40,40]))
        over[mask&~pred]=np.array([40,120,255],np.uint8)
        cv2.drawContours(over,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),1)
        p=OUT/'training_alpha_overlays'/f'{i:02d}.png';p.parent.mkdir(exist_ok=True);Image.fromarray(over).save(p);overlays.append(p)
        np.savez_compressed(OUT/'training_alpha_overlays'/f'{i:02d}_alpha.npz',alpha=alpha)
        scores.append(score)
    current=exp.read(exp.OUT/'evaluation/summary.json')['training_fit_to_generated_inputs']
    groups={}
    for batch in [6,7,8]:
        subset=[r for r in scores if r['batch']==batch];fit=[r for r in current if r['batch']==batch]
        groups[str(batch)]={k:float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in ['index','batch','angle','mask_area','training_outside_opacity_denominator']}
        groups[str(batch)].update({k:float(np.mean([r[k] for r in fit])) for k in ['foreground_psnr','silhouette_iou']})
    shapes={}
    for label,path in [('current24',exp.OUT/'model.ply'),('prior8',exp.prior.OUT/'generated8/model.ply')]:
        v=PlyData.read(path)['vertex'].data
        scales=np.sort(np.exp(np.column_stack([v[f'scale_{j}'] for j in range(3)]).astype(float)),axis=1)
        opacity=1/(1+np.exp(-v['opacity'].astype(float)))
        stats=dict(gaussians=len(v),scale_quantiles=np.quantile(scales,[.5,.9,.99],axis=0).tolist())
        for name,ratio in [('longest_shortest',scales[:,2]/scales[:,0]),('longest_middle',scales[:,2]/scales[:,1]),('middle_shortest',scales[:,1]/scales[:,0])]:
            stats[name]=dict(quantiles=np.quantile(ratio,[.5,.9,.99]).tolist(),opacity_fraction_above4=float(opacity[ratio>4].sum()/opacity.sum()))
        shapes[label]=stats
    prior_rows=exp.read(exp.prior.OUT/'generated8/inputs.json')['frames']
    for a,b in zip(rows[:8],prior_rows):
        assert a['sha256']==b['sha256'] and a['mask_sha256']==b['mask_sha256']
        np.testing.assert_array_equal(a['w2c'],b['w2c']);np.testing.assert_array_equal(a['k'],b['k'])
    videos={}
    for name,expected in [('supervised_arc',128),('boundary67',64),('boundary78',64),('elevated_arc',128),('orbit360',128),('training_comparison',24)]:
        cap=cv2.VideoCapture(str(exp.OUT/'evaluation'/f'{name}.mp4'));count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==expected
        videos[name]=dict(frames=count,sha256=exp.chain.digest(exp.OUT/'evaluation'/f'{name}.mp4'))
    # Same pixel crop for every frame. No object alignment, scale fitting or cleanup.
    canvas=Image.new('RGB',(1536,1560),(22,22,22));draw=ImageDraw.Draw(canvas)
    for row,i in enumerate([7,8,15,16,22,23]):
        gt=np.asarray(Image.open(rows[i]['image']).convert('RGB'));mask=np.asarray(Image.open(rows[i]['mask']))>127
        target=Image.fromarray(gt*mask[:,:,None]);own=Image.open(exp.OUT/'evaluation/training_comparison'/f'{i:03d}.png').crop((832,32,1664,512));over=Image.open(overlays[i])
        for col,(image,label) in enumerate([(target,'Generated target'),(own,'3DGS at target camera'),(over,'Red:outside alpha>0.5 | Green:target contour')]):
            crop=image.crop((144,112,656,344));canvas.paste(crop,(512*col,260*row+28));draw.text((512*col+5,260*row+7),f"{label} | B{rows[i]['batch']} | {rows[i]['angle']:.2f}",fill='white')
    canvas.save(OUT/'training_failure_examples.jpg',quality=96)
    exp.chain.montage(overlays,[f"B{r['batch']} | {r['angle']:.2f}deg | out(alpha>.5)={r['outside_area_over_mask_0.5']:.1%}" for r in scores],OUT/'alpha_review24.jpg',4,(416,240))
    exp.save(OUT/'diagnostics.json',dict(scope='Generated training targets and masks, not realGT; read-only checkpoint rendering',
        by_batch=groups,per_frame=scores,gaussian_shape=shapes,videos=videos,prior8_same_first8_inputs_cameras_masks=True))
    exp.validate();print('ANALYZED',OUT,flush=True)
    print('GROUPS',groups,flush=True)
    print('LAST',scores[-1],flush=True)


if __name__=='__main__':main()
