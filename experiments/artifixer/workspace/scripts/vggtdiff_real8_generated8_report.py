"""Package results of the controlled real8/generated8 reconstruction experiment."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import vggtdiff_real8_generated8_reconstruct as exp


def main():
    root=exp.OUT; summary=exp.ab.read(root/'evaluation/summary.json'); exp.validate()
    for r in exp.ab.read(root/'implementation/manifest.json')['files']:
        assert exp.ab.digest(Path(r['path']))==r['sha256'],r['path']
    exp.ab.frozen_copy(Path(__file__),root/'implementation'/Path(__file__).name)
    refs=exp.ab.read(exp.MATCH/'references.json')['frames']
    refs=[r for r in refs if r['original_frame'] in [960,1200]]
    visuals=[]
    for i,row in enumerate(refs):
        canvas=Image.new('RGB',(2496,512),'#181818'); draw=ImageDraw.Draw(canvas)
        for j,label in enumerate(['Real image','real8 reconstruction','generated8 reconstruction']):
            path=(root/'evaluation/groundtruth/heldout_real_in_arc'/f'{i:03d}.png') if j==0 else (root/['real8','generated8'][j-1]/'evaluation/heldout_real_in_arc'/f'{i:03d}.png')
            canvas.paste(Image.open(path),(832*j,32));draw.text((832*j+8,10),f'{label} | real frame {row["original_frame"]}',fill='white')
        dest=root/'evaluation/heldout_gt_real_generated'/f'{i:03d}.png'; dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest);visuals.append(dest)
    exp.ab.montage(visuals,[f'real frame {r["original_frame"]}' for r in refs],root/'evaluation/heldout_gt_real_generated.jpg',1,(1248,256))
    # Store the exact GT masks with each experiment's diagnostic package.
    for i,row in enumerate(refs): exp.ab.frozen_copy(Path(row['mask']),root/'evaluation/heldout_gt_real_generated'/f'{i:03d}_gt_mask.png')
    videos={}
    for category in ['local','elevated','orbit360','local_alpha','elevated_alpha']:
        path=root/'evaluation'/f'{category}_comparison.mp4'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=width,height,nb_read_frames,codec_name','-of','json',str(path)]))
        stream=probe['streams'][0];assert int(stream['nb_read_frames'])==64 and stream['width']==1664 and stream['height']==512
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        videos[category]=dict(path=str(path),sha256=exp.ab.digest(path),**stream)
    for src,name in [('/tmp/real8_reconstruction_train.log','real8_train.log'),('/tmp/generated8_reconstruction_train.log','generated8_train.log'),('/tmp/real8_generated8_evaluation.log','evaluation.log'),('/tmp/real8_generated8_comparison.log','comparison.log')]:
        target=root/'logs'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(src,target)
    real=summary['real8'];gen=summary['generated8']
    def mean(group,b):return summary[b][group]['mean']
    rows=[]
    for label,key,fmt in [('前景 PSNR ↑','foreground_psnr','.2f'),('轮廓 IoU ↑','silhouette_iou','.4f'),('轮廓外面积 / 真实车面积，alpha>0.5 ↓','outside_area_over_gt_at_0.5','.2%'),('轮廓外面积 / 真实车面积，alpha>0.05 ↓','outside_area_over_gt_at_0.05','.2%')]:
        rows.append(f'| {label} | {mean("heldout_real_in_arc","real8")[key]:{fmt}} | {mean("heldout_real_in_arc","generated8")[key]:{fmt}} |')
    text='''# 真实八帧与生成八帧分别重建

## 结论

在这次共同初始化、固定相机、相同优化配方的实验中，八帧覆盖范围内，两组整体稳定，采样渲染没有显示生成组出现明显更多的大范围飘逸。两组均残留轮廓毛边；小幅抬高相机后，车顶、窗沿、轮胎出现薄层和表面破碎，真实组也明显存在。

完整 360° 环视中，两组离开观测侧后都出现严重拉丝、破碎和悬浮状结构。因此这组实验不能支持“完整环视拉丝全部源于生成图跨视角不一致”。覆盖不足及当前表示/初始化/优化的外推局限是共同因素；它也不能排除生成图不一致对其他混合监督实验的影响。

生成组可以较好拟合自己的八张图，但与真实数据仍有轮廓、形状或投影位置差异。下表的轮廓外面积不能全部解释为飘点：较大的车身、局部形变或位置偏差也会增大它。外观 PSNR 同样不是纯几何指标。

## 实验设置

- 真实帧序号：1340、1300、1220、1180、1100、1060、980、910。
- 相机方位角 224.60°–263.91°，跨度仅 39.31°，不是八帧覆盖完整一圈。
- 生成组 seed20261028，按生成前冻结的 <=15° 主集合14对极线误差中位数选出：三个种子分别为0.642650、0.476064、0.416205 px。该种子不是每个三视角指标都最好。真实组对应主指标0.351205 px。
- 输入832×480，每组仅八张训练图，各625次曝光，全部权重1，固定优化种子20261021，均从0训练到5000步，不混入六张NVS参考图，也不加载此前光度训练的warmup。
- 两组使用相同、固定的W2C/K；这些是现有Omega/PnP估计相机，不是独立标定真值。
- 共同初始化：17112个真实参考图Omega表面点和4000个低透明度体积种子，两组初始模型张量哈希相同。此初始化使用过六张真实参考图的几何，因此本实验不是完全仅凭各自八张图从零估计几何，也不是无真实几何先验的生成图重建。
- 沿用现有3DGRUT模型/优化器/增密策略与gsplat 3DGS渲染器；SH上限1，最多100000点。L1/SSIM/轮廓外透明度权重0.8/0.2/0.2，mask膨胀3×3，各向异性惩罚0.03（比例16以上）。两组均用各自已经人工检查过的车辆mask。
- 相同的相机路径：64帧原相机间位置线性插值+姿态SLERP；同路径沿车辆up增加0.025世界单位；按真实八帧中位距离/高度构造64帧完整环视。完整环视是无监督区域压力展示，不是完整车辆准确性评估。

## 相同真实视角上的检查

真实帧960、1200未参与本次八帧光度优化，均处于覆盖角度内。它们曾作为NVS条件和共同点云来源，所以不是对整个流程完全独立的测试集。数值是两张图的逐图指标均值。

| 指标 | 真实八帧重建 | 生成八帧重建 |
|---|---:|---:|
'''+ '\n'.join(rows)+f'''

各自训练图的平均前景PSNR：真实组{real['own_training']['mean']['foreground_psnr']:.2f} dB，生成组{gen['own_training']['mean']['foreground_psnr']:.2f} dB；平均轮廓IoU分别{real['own_training']['mean']['silhouette_iou']:.4f}、{gen['own_training']['mean']['silhouette_iou']:.4f}。这说明两组能拟合各自输入，不代表两者恢复了相同真实几何。

最终高斯数量：真实组93438，生成组95210。各向异性比值P99分别{real['model']['ratio_quantiles'][2]:.3f}、{gen['model']['ratio_quantiles'][2]:.3f}；生成组没有更高的拉长参数尾部。但参数统计不等价于飘逸评分，仍须结合新视角渲染。

## 可视化与复核

所有成对视频左侧真实八帧，右侧生成八帧。主要查看局部绕视和抬高视角，最后看完整环视外推。

- [局部绕视](evaluation/local_comparison.mp4)
- [抬高视角](evaluation/elevated_comparison.mp4)
- [360°外推](evaluation/orbit360_comparison.mp4)
- [局部透明度](evaluation/local_alpha_comparison.mp4)
- [抬高视角透明度](evaluation/elevated_alpha_comparison.mp4)
- [真实原图 / 真实组重建 / 生成组重建](evaluation/heldout_gt_real_generated.jpg)
- [输入图比较](evaluation/input_contact.jpg)
- [完整指标](evaluation/summary.json)
- [公平性与导出审计](evaluation/audit.json)

已经人工检查三个RGB路径各8个均匀采样帧、局部透明度采样图和中部抬高视角近景。对飘逸的描述是这次试验的定性观察，没有宣称所有像素或所有未观测视角通过质量阈值。

相机/输入哈希匹配、两组初始化张量一致、八张图各625次更新、PLY和渲染有限值检查均通过；两组PLY重新载入在四个环视相机的RGB平均绝对误差均为0。五段视频各64帧、1664×512，逐段解码检查通过。训练1000/3000/5000步检查点均保留。本次只运行一个重建优化种子。

## 复现入口

环境`.venv-car15`，实现`scripts/vggtdiff_real8_generated8_reconstruct.py`。依次执行`prepare`、`train --branch real8`、`train --branch generated8`、`evaluate`、`compare`；训练/渲染使用`CUDA_VISIBLE_DEVICES=2`。已存在的冻结实验拒绝直接覆盖，重新实验应使用新输出目录。打包报告使用`scripts/vggtdiff_real8_generated8_report.py`。源代码快照位于`implementation/`，原始日志在`logs/`，每组`model.ply`为最终高斯模型。
'''
    (root/'REPORT.md').write_text(text)
    exp.ab.save(root/'final_audit.json',dict(state='completed',training_steps_per_branch=5000,selected_seed=exp.SELECTED,
        videos=videos,report_sha256=exp.ab.digest(root/'REPORT.md'),reconstruction_audit=exp.ab.read(root/'evaluation/audit.json'),
        visual_review='RGB contacts: local/elevated/orbit360, eight samples each; local alpha contact; elevated032 closeup; heldout-real render contacts.',
        report_script_sha256=exp.ab.digest(Path(__file__))))
    print('REPORT_READY',root/'REPORT.md')

if __name__=='__main__':main()
