# 环境与权重

[返回主说明](../README.md)

## 环境配置

在 Linux 和带 CUDA 的 NVIDIA GPU 上运行。本流程在 Python 3.12、PyTorch 2.11（CUDA 12.8）、gsplat 1.4.0 上跑通。生成阶段默认把约 46 GiB 权重留在 GPU 上；显存更小的卡把 `--vram-limit-gib` 改成 `0`，权重会按块从 CPU 调入。

| 环境 | 用途 | 关键包 |
| --- | --- | --- |
| `.venv-gen` | 生成 80 帧 | 本仓库、`huggingface_hub`、VGGT-Omega |
| `.venv-e1` | E1 重建 | 与生成环境同一套 PyTorch、`gsplat`、3DGRUT 的 Python 依赖 |

两个环境使用同一主版本的 PyTorch 和 CUDA，避免 gsplat 对着另一套 CUDA 重新编译。

#### 安装

```bash
git clone https://github.com/hjc0106/VGGT-Diff.git
cd VGGT-Diff
git checkout vggt-gen-v2

python -m venv .venv-gen
source .venv-gen/bin/activate
pip install -U pip
pip install -e .
pip install -U huggingface_hub
pip install git+https://github.com/facebookresearch/vggt-omega.git@399d4d62935deb71cedb1e1c35b7a90413a6bee4
deactivate

python -m venv .venv-e1
source .venv-e1/bin/activate
pip install -U pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install gsplat==1.4.0
pip install plyfile omegaconf hydra-core addict rich torchmetrics \
  "opencv-python<4.12" imageio numpy pillow tensorboard scikit-learn tqdm polyscope
deactivate
```

3DGRUT 只需要源码在 `--threedgrut` 指向的目录里，该目录下应能看到 `threedgrut/` 和 `threedgut_tracer/`。使用 [3DGRUT](https://github.com/nv-tlabs/3dgrut) 或 ArtiFixer 维护的 3DGRUT 分支都可以。`requirements.txt` 里的 `slangtorch` 和 `fused-ssim` 是原版光线追踪和融合 SSIM，这条 E1 路径不调用它们。

权重：

```bash
source .venv-gen/bin/activate
hf download chenkangjie1123/VGGT-Diff-Trajectory-480x832 \
  vggt-diff.safetensors config.json LICENSE README.md \
  --local-dir checkpoints/VGGT-Diff-Trajectory-480x832
```

另外准备 `vggt_omega_1b_512.pt`。需要先申请 [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) 的访问权限。Wan2.1 可以在第一次生成时自动下载，也可以用 `--base-model-dir` 指向本地的 `Wan2.1-I2V-14B-480P` 目录。

#### 环境测试

生成环境确认 CUDA 和包可以导入：

```bash
source .venv-gen/bin/activate
python -c "import torch; from vggtdiff import VGGTDiff; assert torch.cuda.is_available(); print('gen', torch.__version__, torch.cuda.get_device_name(0))"
python tests/test_reconstruction_recipe.py
```

`tests/test_reconstruction_recipe.py` 不加载模型。它检查 80 帧闭合轨道会留出最后一帧、训练步数是 49375，并用一张假图把数据集准备跑一遍。打印 `OK` 即通过。

重建环境确认 gsplat 和 3DGRUT 可以导入。把下面的路径换成自己的 3DGRUT 检出目录：

```bash
source .venv-e1/bin/activate
export THREEDGRUT=/path/to/3DGRUT
python -c "import sys; sys.path.insert(0, '$THREEDGRUT'); import torch, gsplat, threedgut_tracer, threedgrut.trainer; assert torch.cuda.is_available(); print('e1', torch.__version__, gsplat.__version__)"
```
