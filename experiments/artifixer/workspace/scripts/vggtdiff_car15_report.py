"""Summarize the completed car15 experiment without treating synthetic views as GT."""
from pathlib import Path
from vggtdiff_car15_prepare import OUT,read
from vggtdiff_white_car import save,digest

def main():
    metrics=read(OUT/'evaluation/metrics.json');stages=metrics['stages']
    lines=['# 白车十五图重建实验','', '输入为六张真实参考图与九张 VGGT-Diff 生成图。真实参考帧对应原视频 0、480、720、960、1200、1440；生成图目标角度为 45° 至 135°，间隔 11.25°。', '', '两条分支共享六张真实图训练至 1000 步的检查点，分别继续至总计 5000 步：real6 为仅真实图基线，mixed15 为六张真实图加九张生成图。mixed15 按真实/生成两组各 50% 概率抽样，生成图损失权重为 0.3。所有相机固定，按各自内参使用原始 832×480 分辨率。','', 'SAM3 提取车身及轮胎，背景透明；损失包括前景归一化 RGB、车体包围区域 masked SSIM 和掩码外不透明度抑制。初始化使用 17112 个真实表面点和 4000 个低透明度体积种子；SH 最高 1 阶，高斯数量上限 100000。', '', '**实现变更：** 原生 3DGUT Slang 内核因编译器版本兼容性无法运行，本实验采用 gsplat 标准 3DGS 光栅化后端，保留 3DGRUT 的高斯模型、优化器、增密和导出。两条分支使用相同后端。这是 VGGT-Diff 图像辅助重建实验，并非 VGGT-Diff 原生直接输出 3DGS。', '', '## 可视检查结论', '', '已检查 64 帧全环绕总览、45°/90°/135°关键视角、六个抬高视角和两张真实验证图。mixed15 在生成图覆盖的右侧 45°–135°形成了明显更完整的车门、车窗、轮胎和侧面轮廓；real6 在该侧呈现大片破碎高斯。真实拍摄侧的外观变化较小。', '', '正前 0°、正后 180°以及车顶、轮廓边缘仍有拉丝、破碎和漂浮高斯；较高观察点也可见结构不稳定。九张生成图集中于右侧，不能弥补所有角度的观测不足。它是隐藏侧外观补充的成功初步实验，尚未达到干净完整车体重建。', '', '真实训练图拟合下降；独立真实帧 PSNR 提升约 0.19 dB，但轮廓 IoU 从 0.9348 降至 0.9137，不能声称整体质量全面提高。生成图分数的大幅提升属于对合成监督的拟合。', '', '## 数值结果', '', 'PSNR/MAE 只在白车掩码内计算；IoU 比较 alpha>0.5 与分割掩码。训练拟合与独立验证分开列示。', '', '| 数据 | 分支 | 前景 PSNR ↑ | 前景 MAE ↓ | 轮廓 IoU ↑ |', '|---|---|---:|---:|---:|']
    for group,label in [('real_training','真实训练六图'),('generated_training','生成训练九图（拟合分数）'),('independent_real','独立真实验证两图')]:
        for stage in ['real6','mixed15']:
            x=stages[stage]['aggregate'][group];lines.append(f"| {label} | {stage} | {x['foreground_psnr']:.3f} | {x['foreground_mae']:.5f} | {x['silhouette_iou']:.4f} |")
    lines+=['', '独立验证帧为原视频 600 和 1080，未参与条件参考或重建训练。其相机由 SIFT + PnP 对齐至原 Omega 世界坐标，分别有 16、23 个内点，1280 尺寸下中位重投影误差约 0.923、0.725 像素。位姿仍存在估计误差；这些帧只验证真实拍摄侧，不能验证未拍摄侧。', '', '## 交付文件', '']
    for stage in ['real6','mixed15']:
        r=read(OUT/'training'/f'{stage}_result.json');lines += [f"- `{stage}.ply`：{stages[stage]['gaussian_count']} 个高斯，包含 opacity、scale、rotation 和 SH 属性；所有属性已检查为有限值。",f"- 检查点：`{r['checkpoint']}`。"]
    lines += ['- `evaluation/mixed15_orbit360.mp4`：十五图模型单独的 64 帧环绕视频。','- `evaluation/orbit360_comparison.mp4`：64 帧完整环绕，左 real6、右 mixed15，8 fps。','- `evaluation/orbit_contact.jpg` 与 `evaluation/all64_contact.jpg`：关键帧及全部帧总览。','- `evaluation/elevated_contact.jpg`：六个较高观察点的比较，用于暴露平面化和漂浮结构。','- `evaluation/validation_comparison_600.png`、`evaluation/validation_comparison_1080.png`：真实验证图、real6、mixed15 对照。','- `evaluation/metrics.json`：逐图与分组指标；`evaluation/audit.json`：冻结输入/脚本 SHA256 和共享检查点审计。','', '## 复现', '', '在仓库根目录，使用 `.venv-car15/bin/python`；训练脚本为 `scripts/vggtdiff_car15_train.py`，按 warmup、real6、mixed15 顺序运行。评估脚本为 `scripts/vggtdiff_car15_evaluate.py`。CUDA 环境与依赖版本记录于 experiment_manifest.json。数据坐标为原 VGGT-Omega 相对尺度世界坐标，未标定米制尺度。', '', '## 解释边界', '', '生成图与自身的拟合提升只说明能够拟合该合成监督，不能当作隐藏侧重建精度。即使轮廓连贯，反面纹理、部件位置和体积仍可能受生成不一致、稀疏视角及估计相机误差影响。最终质量判断须结合 360° 与抬高相机的可视检查。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'delivery.json',dict(status='completed and visually reviewed',models={s:dict(path=str(OUT/f'{s}.ply'),gaussians=stages[s]['gaussian_count'],sha256=digest(OUT/f'{s}.ply')) for s in stages},report=str(OUT/'REPORT.md')))
if __name__=='__main__':main()
