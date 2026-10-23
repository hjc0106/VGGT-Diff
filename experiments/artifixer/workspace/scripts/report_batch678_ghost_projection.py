"""Summarize fixed-model projection conflicts; no training or model mutation."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
import vggtdiff_batch678_regularization_ablation as study

OUT=study.OUT/'ghost_projection_diagnosis_v1'


def main():
    read=lambda n:json.loads((OUT/n).read_text())
    data=np.load(OUT/'gaussian_attribution.npz');selected=data['selected_ids']
    mass=data['region_mass'][:,selected,:];visible=mass.sum(2)
    inside_fraction=mass[:,:,0]/np.maximum(visible,1e-12)
    outside_fraction=mass[:,:,1]/np.maximum(visible,1e-12)
    sensitivity=[];default=None
    for absolute in [.01,.1,1.]:
        for relative in [.01,.05,.1]:
            valid=visible>=np.maximum(absolute,relative*visible.max(0))[None,:]
            supported=(inside_fraction>.5)&valid;outside=(outside_fraction>.5)&valid
            mixed=supported.any(0)&outside.any(0)
            r=dict(min_visible_mass=absolute,min_relative_to_peak=relative,
                mixed=int(mixed.sum()),inside_any=int(supported.any(0).sum()),outside_any=int(outside.any(0).sum()),
                outside_without_inside=int((outside.any(0)&~supported.any(0)).sum()),
                anchor_selected_mass_in_mixed=float(data['anchor_mass'][selected[mixed],0].sum()/data['anchor_mass'][selected,0].sum()))
            sensitivity.append(r)
            if absolute==.1 and relative==.05:
                default=r
                np.savez_compressed(OUT/'conflict_ids.npz',mixed=selected[mixed],
                    outside_without_inside=selected[outside.any(0)&~supported.any(0)],
                    selected_ids=selected,supported_views=supported,outside_views=outside,visible_views=valid)
    study.base.save(OUT/'conflict_classification.json',dict(default=default,sensitivity=sensitivity,
        rule='Visible mass >=max(absolute threshold, relative*Gaussian peak visibility across24); support/conflict means >50% of visible contribution in eroded interior/dilated exterior. Exploratory, not a pruning rule.',
        caution='Inside mask is silhouette compatibility, not proof of correct depth or target appearance; 24 nearby views are not independent.'))
    s=read('summary.json');selection=read('selection.json');g=read('group_view_contributions.json')
    table=[]
    for group in ['upper_outline','middle_outline','lower_outline']:
        allg=[v for v in g if v['group']==group]
        row=dict(group=group,gaussians=selection['groups'][group])
        for b in [6,7,8]:
            sub=allg[(b-6)*8:(b-5)*8];total=sum(v['visible_mass'] for v in sub)
            row[f'batch{b}_inside']=sum(v['inside_mass'] for v in sub)/total
            row[f'batch{b}_outside']=sum(v['outside_mass'] for v in sub)/total
        row['endpoint_outside']=allg[-1]['outside_fraction'];row['endpoint_visible_to_isolated']=allg[-1]['visible_to_isolated'];table.append(row)
    study.base.save(OUT/'group_summary.json',table)
    # Fixed crop of the existing contact panels, magnified for inspection only.
    labels=['Generated target','Original B','Selected contribution','Remove selected (diagnostic)']
    for index in [22,23]:
        source=Image.open(OUT/'training_panels'/f'{index:02d}.jpg')
        result=Image.new('RGB',(2048,260),(20,20,20));draw=ImageDraw.Draw(result)
        for i,label in enumerate(labels):
            crop=source.crop((416*i+72,28+56,416*i+328,28+172)).resize((512,232))
            result.paste(crop,(512*i,28));draw.text((512*i+5,8),label,fill='white')
        result.save(OUT/f'endpoint{index}_zoom.jpg',quality=96)
    baseline=s['original'];union=s['all_selected']
    lines=['# 末端拖影高斯：跨24监督视角归因及临时移除诊断','',
        '对象：固定24图正则消融中的B轮廓约束模型，87770个高斯。训练、模型、输入图、掩膜和相机均未修改；不导出删除后的生产模型。','',
        '## 核心结论','',
        f'两个末端视角348.75°/354.375°中，3270个高斯（{3270/87770:.2%}）解释了扩张掩膜外且alpha>.05区域90.003%的alpha合成贡献。按末端投影贡献的纵向位置分为上部1015、中部1842、下部413，分组是图像位置，不是严格语义车体分割。','',
        f'按明确的探索性可见性规则，{default["mixed"]}/3270（{default["mixed"]/3270:.1%}）个高斯在某些视角主要贡献于车内，在另一些视角主要贡献于轮廓外。这批高斯占选中高斯末端归因质量的{default["anchor_selected_mass_in_mixed"]:.1%}。因此主要残留高斯与正常车体拟合交织，不能把候选集整体当作可直接删除的浮点。','',
        '这一结果确认了当前模型中的跨视角轮廓兼容冲突，尚未证明生成图像彼此不可能几何一致。错误的高斯中心/尺度/深度、欠约束、掩膜偏差或相机偏差都可能导致这种结果。','',
        '## 可见贡献如何随视角变化','',
        '| 高斯组 | 数量 | 第6批车内贡献占比 | 第7批车内贡献占比 | 354.375°轮廓外占比 | 末端可见/单独渲染 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f'| {r["group"]} | {r["gaussians"]} | {r["batch6_inside"]:.1%} | {r["batch7_inside"]:.1%} | {r["endpoint_outside"]:.1%} | {r["endpoint_visible_to_isolated"]:.1%} |')
    lines+=['','车内/车外比例使用真实前后遮挡下的合成贡献；末端可见/单独渲染比较只用于检查遮挡。第6/7批高斯组主要投影在车体内部，正面末端则大量越出轮廓；它们并非在其余视角完全不可见。','',
        '## 临时移除：改善与代价','',
        '| 暂时置零的组 | 末端轮廓外面积alpha>.5 | 24图平均PSNR | PSNR相对B变化 | 平均缺失alpha>.5 |',
        '|---|---:|---:|---:|---:|']
    for label,r in s.items():lines.append(f'| {label} | {r["endpoint"]["outside_0.5"]:.2%} | {r["foreground_psnr"]:.2f}dB | {r["foreground_psnr"]-baseline["foreground_psnr"]:+.2f}dB | {r["missing_0.5"]:.3%} |')
    lines+=['',
        f'整批移除使末端外面积从{baseline["endpoint"]["outside_0.5"]:.2%}降到{union["endpoint"]["outside_0.5"]:.2%}，但24图PSNR下降{baseline["foreground_psnr"]-union["foreground_psnr"]:.2f}dB，平均缺失从{baseline["missing_0.5"]:.3%}增到{union["missing_0.5"]:.3%}。最严重的颜色拟合下降在348.75°，降低12.86dB；图中窗框、边缘及车体局部发生变化。不能因轮廓IoU提高就接受删除。','',
        '选中高斯解释90%当前可见轮廓外alpha贡献，移除后外部alpha积分只下降约一半。这不矛盾：临时移除会改变遮挡，后方原先被挡住的高斯会继续贡献拖影；面积阈值指标本身也不是线性贡献积分。','',
        f'另有{default["outside_without_inside"]}个高斯在当前阈值下曾主要投影于背景、从未主要投影于车内，可列为进一步精查候选；这不等于没有任何车内贡献，也不构成自动删除依据。其移除收益本轮未单独验证。','',
        '## 方法与验证','',
        '归因等于每高斯每像素的T_i*alpha_i合成权重求和，不使用中心点落入掩膜代替可见性。CUDA路径使用对诊断颜色求导的恒等式；实际执行由于CUDA驱动初始化失败，使用安装版本gsplat的CPU投影参考及等价C++分块合成直接累加权重。保留tile16、eps2d=.3、near=.001、alpha截断.999、最低alpha1/255、透射率1e-4提前终止规则。','',
        'CPU结果在全部24相机上与原CUDA保存的浮点alpha及PNG颜色比较，通过后才继续；精确误差见audit.json。颜色对照PNG包含8位量化误差，面积指标对照未量化alpha。各高斯区域贡献和与alpha区域积分吻合，组标记渲染与逐高斯贡献和吻合。','',
        '车内区域用3×3腐蚀掩膜，车外用3×3扩张掩膜之外，中间作为边界不确定带；面积/缺失指标仍用原始掩膜。候选排序按两锚视角可见质量合计，不等权平均两图。','',
        '分类规则：高斯在一视角的可见贡献至少0.1 alpha-pixel，且至少达到其24视角峰值5%；车内/外贡献超过该视角可见贡献50%分别记为支持/冲突。参数敏感性以0.01/0.1/1像素和峰值1%/5%/10%的9种组合另存。它们是本轮诊断规则，不是几何合格线。','',
        f'9种阈值组合下，mixed数量为{min(r["mixed"] for r in sensitivity)}至{max(r["mixed"] for r in sensitivity)}，对应选中高斯锚视角贡献质量的{min(r["anchor_selected_mass_in_mixed"] for r in sensitivity):.1%}至{max(r["anchor_selected_mass_in_mixed"] for r in sensitivity):.1%}。因此83.5%依赖指定诊断阈值，不能当作模型固有的冲突概率。可见贡献热力图和完整移除的拟合代价不依赖此分类阈值。','',
        '## 下一步建议','',
        '优先检查mixed高斯的中心和长轴在相邻末端视角中的位置，以及对应监督边缘/轮廓是否要求互相冲突的投影。将“模型自身错误几何”与“监督或相机不一致”区分后，再设置重建约束或补可靠视角。当前证据支持避免整批硬删，也不足以直接决定应增加多少图或归责于某一帧。','',
        '## 文件','',
        '- conflict_heatmap.png：三组×24相机的车外占比、车内占比、遮挡比例。',
        '- endpoint22_zoom.jpg / endpoint23_zoom.jpg：固定区域放大，仅用于观看。',
        '- training_panels/、group_removals/：24图高亮与各组临时移除。',
        '- *_removal_comparison.mp4：全部五轨迹，左原B，右临时移除候选并集。',
        '- gaussian_attribution.npz / conflict_ids.npz：原checkpoint编号、逐视角贡献及分类。',
        '- per_view_metrics.csv / group_view_contributions.csv：逐视角原始数值。',
        '- protocol.json / selection.json / audit.json：方法、选择依据和运行检查。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('CLASSIFICATION',default,'sensitivity mixed range',min(r['mixed'] for r in sensitivity),max(r['mixed'] for r in sensitivity),flush=True)


if __name__=='__main__':main()
