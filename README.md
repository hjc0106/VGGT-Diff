# VGGT-Diff · 白车 E1 复现

`vggt-gen-v1` 在 VGGT-Diff 上封装了 80 帧 Trajectory 生成与物体 3DGS 重建，并提供旧白车 E1 实验的相机配置、输入校验清单、复现入口和验收结果。

上游研究与模型归原作者所有：[原项目](https://github.com/chenkangjie1123/VGGT-Diff) · [项目主页](https://chenkangjie1123.github.io/VGGT-Diff/) · [论文](https://arxiv.org/abs/2609.33253)。

## 结果预览

![白车 E1 复现的八视角检查图](results/white_car_e1_v1/orbit_contact.jpg)

使用旧实验相同的 160 个独立环绕相机渲染，上图等间隔选取八个视角。仓库仅保留这张检查图与[逐视角指标 JSON](results/white_car_e1_v1/benchmark.json)。

| 指标（中位数） | 旧 E1 | 本分支实测 |
| --- | ---: | ---: |
| 79 个训练视角前景 PSNR | 24.3939 dB | 24.2480 dB |
| 训练轮廓 IoU | 0.90585 | 0.90367 |
| 16 张独立真实图前景 PSNR | 17.7400 dB | 17.7498 dB |
| 独立真实图轮廓 IoU | 0.88703 | 0.88629 |

训练图、掩码、相机和初始化全部使用旧实验固定数据。训练集与真实留出集的 PSNR、IoU 和轮廓外不透明度共六项检查均通过预设容差。复现结果允许数值差异；未知侧面的真实几何尚无完整真值验证。

[完整验证记录](docs/white_car_e1_v1_validation.md) · [结果文件说明](results/white_car_e1_v1/README.md)

## 快速复现旧 E1

需要 Linux、NVIDIA CUDA GPU、重建环境和 3DGRUT 源码。环境安装步骤见 [环境与权重](docs/setup.md)。

```bash
git clone --branch vggt-gen-v1 https://github.com/hjc0106/VGGT-Diff.git
cd VGGT-Diff

# 仅检查固定输入，无需 GPU
.venv-e1/bin/python scripts/reproduce_white_car_e1.py --example /path/to/white_car_fixture --verify-only

# 从固定输入重新训练并验收
CUDA_VISIBLE_DEVICES=0 .venv-e1/bin/python scripts/reproduce_white_car_e1.py \
  --example /path/to/white_car_fixture \
  --threedgrut /path/to/3DGRUT \
  --output outputs/white_car_e1_v1
```

完整固定输入未上传，需通过 `--example` 指定本地完整基准目录；目录结构与 SHA-256 见 [白车说明](examples/white_car/README.md)。运行前会校验 204 个文件，直接重建无需扩散模型权重。

流程使用 79 个不同视角，每图更新 625 次，共 49,375 步；第 80 帧是首尾同相机复测，不参与训练。初始化为 17,112 个观测表面点与 4,000 个低不透明度体积种子。在 RTX A6000 上重建约需 21 分钟。

输出包含 `e1/model.ply`、`evaluation/benchmark.json`、`evaluation/orbit360.mp4` 和检查图。输入发生变化会拒绝复用旧数据；训练中断后请使用新输出目录。

## 重新生成与新场景

- **检查生成入口**：使用固定六张参考图和原轨迹重新生成 80 帧，命令见 [白车说明](examples/white_car/README.md)。本次重新生成与旧 PNG 的全图 PSNR 中位数为 49.09 dB，存在小幅像素差异；上述 E1 指标来自固定旧图。新生成图需要自己的分割掩码。
- **从视频开始**：`gen_pro` 保留选帧、虚拟相机重投影、点云和轨迹构建，见 [视频预处理](docs/video_pipeline.md)。这些步骤会建立新的实验条件，尚未纳入固定旧 E1 的复现验收。
- **其他推理与训练**：见 [入口与参数说明](docs/upstream_usage.md)。

目标相机统一使用 OpenCV W2C，内参使用图像分辨率下的像素单位。源图、位姿和内参需要成套保持一致。

## 验证

```bash
python -m pytest tests -q
```

已完成 37 项测试、wheel 构建、49,375 步实际重训，以及原 16 张真实留出图和 160 机位环绕评估。机器可读运行记录见 [validation.json](docs/white_car_e1_v1_validation.json)。

## 引用与许可

```bibtex
@article{chen2026vggtdiff,
  title={VGGT-Diff: Visual Geometry Meets Diffusion for Sparse-View Novel View Synthesis},
  author={Chen, Kangjie and Li, Xiangyu and Zhang, Dongbin and Zheng, Chaoda and Chen, Shijia and Deng, Jinhao and Lin, Hongbin and Choo, Sin Wai and Wang, Minqi and Yang, Minghao and Zhong, Dake and Song, Guorui and Zhang, Yu and Liu, Xianming and Wang, Boyang},
  journal={arXiv preprint arXiv:2609.33253},
  year={2026}
}
```

本项目基于 [Wan2.1](https://github.com/Wan-Video/Wan2.1)、[VGGT-Omega](https://github.com/facebookresearch/vggt-omega)、[FrameCrafter](https://github.com/szqwu/FrameCrafter) 与 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)。

VGGT-Diff 使用 [VGGT-Diff Research License](LICENSE)，限非商业研究和教育用途。第三方组件遵循各自许可，见 [NOTICE](NOTICE)。
