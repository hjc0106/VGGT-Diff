"""Diagnose completed ablation models and compare fixed-camera artifacts."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
import vggtdiff_batch678_regularization_ablation as study


def diagnose(branch):
    study.base.OUT=study.OUT/branch
    import analyze_batch678_reconstruction as analysis
    analysis.OUT=study.base.OUT/'evaluation/analysis_v1'
    analysis.main()


def compare():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roots={'A_baseline':study.BASELINE,'B_outline':study.OUT/'outline','C_needle':study.OUT/'needle'}
    if (study.OUT/'combined/evaluation/analysis_v1/diagnostics.json').exists():roots['D_combined']=study.OUT/'combined'
    dest=study.OUT/'comparison';dest.mkdir(exist_ok=True)
    metrics={}
    base_initial=study.base.read(study.BASELINE/'training/initial_state.json')['sha256']
    for label,root in roots.items():
        study.base.OUT=root;study.base.validate()
        for name in ['inputs.json','sampling_plan.json','evaluation/cameras.json','shared/initial_points.ply']:
            assert study.base.chain.digest(root/name)==study.base.chain.digest(study.BASELINE/name),name
        r=study.base.read(root/'training/result.json');assert r['initial_state_sha256']==base_initial
        assert set(r['actual_sample_counts'].values())=={625} and len(r['actual_sample_counts'])==24
        d=study.base.read(root/'evaluation/analysis_v1/diagnostics.json')
        fit=study.base.read(root/'evaluation/summary.json')['training_fit_to_generated_inputs'];per=d['per_frame']
        m=dict(mean_psnr=float(np.mean([f['foreground_psnr'] for f in fit])),
            mean_mae=float(np.mean([f['foreground_mae'] for f in fit])),mean_iou=float(np.mean([f['silhouette_iou'] for f in fit])),
            mean_outside05=float(np.mean([f['outside_area_over_mask_0.5'] for f in per])),
            mean_outside005=float(np.mean([f['outside_area_over_mask_0.05'] for f in per])),
            mean_missing05=float(np.mean([f['missing_mask_fraction_0.5'] for f in per])),
            mean_outside_opacity=float(np.mean([f['outside_opacity_over_mask_area'] for f in per])),
            endpoint_outside05=per[-1]['outside_area_over_mask_0.5'],endpoint_outside005=per[-1]['outside_area_over_mask_0.05'],
            endpoint_missing05=per[-1]['missing_mask_fraction_0.5'],endpoint_iou=per[-1]['iou_0.5'],
            endpoint_psnr=fit[-1]['foreground_psnr'],gaussians=r['gaussians'],
            longest_middle_median=d['gaussian_shape']['current24']['longest_middle']['quantiles'][0],
            longest_shortest_median=d['gaussian_shape']['current24']['longest_shortest']['quantiles'][0])
        metrics[label]=m
    baseline=metrics['A_baseline'];decisions={}
    for label,m in metrics.items():
        if label=='A_baseline':continue
        safe=m['mean_psnr']>=baseline['mean_psnr']-1 and m['mean_missing05']<=baseline['mean_missing05']+.005
        gain=(m['endpoint_outside05']<=.75*baseline['endpoint_outside05'] or m['mean_outside05']<=.8*baseline['mean_outside05']) and m['mean_outside_opacity']<=1.05*baseline['mean_outside_opacity']
        decisions[label]=dict(numeric_safety=safe,useful_gain=gain,endpoint_goal_met=m['endpoint_outside05']<=.08,
            psnr_delta_db=m['mean_psnr']-baseline['mean_psnr'],missing_delta=m['mean_missing05']-baseline['mean_missing05'],
            mean_outside05_reduction=1-m['mean_outside05']/baseline['mean_outside05'],endpoint_outside05_reduction=1-m['endpoint_outside05']/baseline['endpoint_outside05'])
    study.base.save(dest/'summary.json',dict(metrics=metrics,decisions=decisions,
        rule=study.base.read(study.OUT/'decision_rule.json'),same_initial_state=base_initial,
        caution='Single reconstruction seed, generated-target masks, no independent realGT; numeric screen still requires visual review'))
    n=len(roots);canvas=Image.new('RGB',(512*(n+1),260*6),(22,22,22));draw=ImageDraw.Draw(canvas)
    rows=study.base.read(study.BASELINE/'inputs.json')['frames']
    for row,i in enumerate([7,8,15,16,22,23]):
        gt=np.asarray(Image.open(rows[i]['image']).convert('RGB'));mask=np.asarray(Image.open(rows[i]['mask']))>127
        target=Image.fromarray(gt*mask[:,:,None]).crop((144,112,656,344));canvas.paste(target,(0,260*row+28));draw.text((5,260*row+7),f'Input | {rows[i]["angle"]:.2f}deg',fill='white')
        for col,(label,root) in enumerate(roots.items(),1):
            im=Image.open(root/'evaluation/training_comparison'/f'{i:03d}.png').crop((832,32,1664,512)).crop((144,112,656,344))
            canvas.paste(im,(512*col,260*row+28));draw.text((512*col+5,260*row+7),label,fill='white')
    canvas.save(dest/'training_examples.jpg',quality=96)
    canvas.crop((0,1300,512*(n+1),1560)).save(dest/'endpoint354_comparison.jpg',quality=96)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained');labels=list(metrics)
    for ax,key,title,mult in [(axes[0],'endpoint_outside05','Endpoint354 outside area, %',100),(axes[1],'mean_psnr','All24 foreground PSNR, dB',1),(axes[2],'mean_missing05','All24 missing foreground, %',100)]:
        vals=[metrics[k][key]*mult for k in labels];bars=ax.bar(labels,vals,color=['#777777','#3979b9','#d66140','#3c9b62'][:n]);ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_title(title);ax.tick_params(axis='x',rotation=20);ax.margins(y=.2);ax.grid(axis='y',alpha=.2)
    fig.savefig(dest/'metrics.png',dpi=160);plt.close(fig)
    # All videos use the identical frozen camera arrays; no per-object fitting.
    for category in ['supervised_arc','boundary67','boundary78','elevated_arc','orbit360']:
        files=sorted((study.BASELINE/'evaluation'/category).glob('*.png'))
        video=dest/f'{category}_comparison.mp4'
        with imageio.get_writer(str(video),fps=12,macro_block_size=16) as writer:
            for i,p in enumerate(files):
                panel=Image.new('RGB',(832*n,512),(22,22,22));d=ImageDraw.Draw(panel)
                for col,(label,root) in enumerate(roots.items()):
                    im=Image.open(root/'evaluation'/category/p.name).convert('RGB');panel.paste(im,(832*col,32));d.text((832*col+8,10),f'{label} | {category} | {i:03d}',fill='white')
                writer.append_data(np.asarray(panel))
        print('COMPARED',category,flush=True)
    # Human-readable table, rewritten after visual interpretation by the caller.
    lines=['# 固定24图：轮廓/针状高斯正则消融','',
        'A复用原24图基线，B仅更改轮廓外项，C仅新增针状项。全部图像、掩膜、W2C/K、初始化张量、采样顺序、训练15000步/每图625次更新、优化器与渲染相机一致。原始数据和基线没有修改。','',
        '| 指标 | '+' | '.join(labels)+' |','|---|'+'---:|'*n]
    for title,key,mult,fmt in [('354.375°轮廓外面积alpha>.5，%','endpoint_outside05',100,'.2f'),('354.375°轮廓外面积alpha>.05，%','endpoint_outside005',100,'.2f'),('全部24图轮廓外面积alpha>.5，%','mean_outside05',100,'.2f'),('全部24图轮廓外面积alpha>.05，%','mean_outside005',100,'.2f'),('全部24图轮廓内缺失alpha>.5，%','mean_missing05',100,'.3f'),('前景PSNR均值dB','mean_psnr',1,'.2f'),('轮廓IoU均值','mean_iou',1,'.4f'),('最长轴/中间轴中位','longest_middle_median',1,'.3f'),('最终高斯数','gaussians',1,'.0f')]:
        lines.append('| '+title+' | '+' | '.join(format(metrics[k][key]*mult,fmt) for k in labels)+' |')
    lines += ['', '## 预先固定的设置与判读','',
        'B：轮廓外alpha积分除以车辆掩膜面积，权重0.05；保留3×3扩张。C：保留基线全部项，新增0.01×透明度加权relu(log(smax/smid)-log4)。基线长短轴比>16的0.03惩罚不变。','',
        '工程目标：正面轮廓外面积约8%以内；所有24图平均PSNR降低≤1dB，车内缺失率增加≤0.5个百分点。组合组条件在结果产生前写入decision_rule.json。它们是本轮筛选目标，不是通用几何质量标准。','',
        '所有数值来自生成训练图与mask，不是独立真实GT；单训练种子。alpha>.05、透明度积分和缺失率同时报告，避免只降低透明度或削掉车体得到较好的单一指标。360°仍含未覆盖侧，不能把外推伪影消失作为当前覆盖下的必然结果。','',
        '## 文件','',
        '- endpoint354_comparison.jpg、training_examples.jpg：固定像素裁剪，无逐图配准/缩放/清理。',
        '- *_comparison.mp4：相同相机、左右顺序按表头的全部5条轨迹。',
        '- summary.json：指标、数值筛选结果与公平性检查。',
        '- 各组evaluation/analysis_v1/diagnostics.json：逐图轮廓与模型尺度数据。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('DECISIONS',decisions,flush=True);print('METRICS',metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnose','compare']);p.add_argument('--branch',choices=['outline','needle','combined']);a=p.parse_args()
    if a.action=='diagnose':
        if not a.branch:p.error('--branch required')
        diagnose(a.branch)
    else:compare()
