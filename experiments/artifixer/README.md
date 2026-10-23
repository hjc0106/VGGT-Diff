# ArtiFixer 白车 VGGT-Diff 实验与跨服务器迁移

这是研究实验代码归档，保留目标仓库原有内容。`workspace/`复现原ArtiFixer目录布局；`workspace/thirdparty/VGGT-Diff/`包含实际使用的推理实现、NVS修复与上游许可证。原始服务器上的脚本、数据及冻结实验没有改动。

## 覆盖范围

| 实验阶段 | 主要入口（位于workspace/scripts） |
|---|---|
| 基础生成、噪声/上下文/分组控制 | vggtdiff_white_car.py、vggtdiff_factorial.py、vggtdiff_noise_control.py、vggtdiff_controlled_context.py、vggtdiff_shared_context_short.py |
| 实拍参考、隐藏侧与整圈64视角 | vggtdiff_realvideo_unseen.py、vggtdiff_realvideo360*.py、vggtdiff_realvideo64.py、vggtdiff_orbit360_latest_refs.py |
| 实拍相机、SfM对齐、留出GT | vggtdiff_prepare_real_targets.py、vggtdiff_real_front_*.py、audit_realvideo_*.py |
| 同相机三种子几何评估 | vggtdiff_matched_gt.py、crossview_geometry_eval.py、crossview_geometry_dense_gt.py、crossview_geometry_matched_eval.py、vggtdiff_matched_spatial_diagnostics.py |
| 第6→7→8批与双侧补间 | vggtdiff_generated_batch_chain.py、vggtdiff_batch07_continuous.py、vggtdiff_batch08_*.py、vggtdiff_batch06_08_to_batch07_continuous.py、crossview_geometry_batch67.py、evaluate_batch07_two_sided_bridge.py |
| 真实/生成8图及15图3DGS对照 | vggtdiff_real8_generated8_reconstruct.py、vggtdiff_car15_*.py |
| 监督覆盖、深度、自洽与正则消融 | vggtdiff_latest_coverage_*.py、vggtdiff_selfconsistent_silhouette.py、vggtdiff_batch678_reconstruct.py、vggtdiff_batch678_regularization_ablation.py |
| 拖影定位、临时移除与CPU参考光栅器 | diagnose_batch678_ghost_gaussians.py、ghost_cpu_reference.py、ghost_cpu_rasterizer.cpp、report_batch678_ghost_projection.py |
| 当前16相机A/B/C三层评估 | vggtdiff_consistency_abc.py |
| 报告、可视化与ArtiFixer二维修复接入 | 对应report/review/build脚本、vggtdiff_car15_artifixer.py |

`SOURCE_INDEX.json`逐项记录原文件相对路径、源文件SHA256、归档SHA256与模型路径适配情况。`DEPENDENCIES.json`记录外部依赖的确切提交及补丁；`dependency_patches/`包括本机3DGRUT修改和COLMAP导出修复。所有测试及静态引用依赖一并归档。

## 新服务器环境

安装适合目标驱动的CUDA版PyTorch/torchvision和ffmpeg；重建/CPU光栅器还需要C++编译器。以下从本目录执行，使用Python3.10以上：

```bash
python -m venv .venv
source .venv/bin/activate
# 先安装与新服务器CUDA匹配的torch/torchvision，再安装：
pip install -e workspace/thirdparty/VGGT-Diff
pip install -r requirements-eval.txt
```

分割需安装SAM3及其依赖；Omega推理需VGGT-Ω。3DGS对照额外需要3DGRUT-ArtiFixer、gsplat及其CUDA编译依赖。按`DEPENDENCIES.json`中的URL克隆到`workspace/thirdparty/<name>`，checkout指定commit，并在依赖仓库中执行`git apply <本目录/dependency_patches/对应补丁>`。补丁不为空时必须应用。对这些项目按各自安装说明安装，模型权重单独下载/复制。

归档中的ArtiFixer二维修复、Docker工作流只是接入层，仍需要完整ArtiFixer服务/模型训练仓库；不属于独立VGGT-Diff推理环境。运行这些集成时设置`ARTIFIXER_ROOT`为完整原项目的迁移位置。普通生成、几何评估和3DGS对照使用本归档workspace。

本机软件版本快照见`ENVIRONMENT_REFERENCE.json`，用于追踪环境，不要求在新硬件上盲目锁定相同CUDA构建版本。

配置模型路径（原机绝对路径已在归档副本中替换为可配置入口）：

```bash
export VGGTDIFF_MODEL_ROOT=/your/models
export VGGTDIFF_CHECKPOINT=/your/models/vggt-diff/vggt-diff.safetensors
export VGGTDIFF_OMEGA=/your/models/vggt-omega/vggt_omega_1b_512.pt
export VGGTDIFF_BASE_MODEL=/your/models/Wan2.1-I2V-14B-480P
```

SAM3权重默认位于`workspace/thirdparty/sam3/checkpoints/sam3/sam3.1_multiplex.pt`。本地纯CPU分割适配包含可选位置缓存与页锁定兼容处理，不更改SAM3源码。

## 数据单独迁移

Git不包含私有视频、图像、相机数据、模型权重、检查点、环境或大体积结果。复制代码不等于已经复制实验输入。

可在旧服务器生成待传输文件清单；它会包含所选实验目录以及JSON引用的其他pipeline_exp输入：

```bash
python asset_inventory.py --workspace /old/ArtiFixer \
  --data-root /old/ArtiFixer/pipeline_exp/realworld/scene-1-nofull/06_worldact_white_car_v1/artifixer_episode_chain_64x4 \
  --output /tmp/white_car_transfer.json
rsync -a --files-from=/tmp/white_car_transfer.files /old/ArtiFixer/ \
  user@new-server:/new/VGGT-Diff/experiments/artifixer/workspace/
```

权重、外部依赖仓库和缓存另外复制。首次`prepare`可能还需要原视频和`3dgs_wcar/colmap/sparse/0`，相应数据应覆盖在传输清单/手工复制中。执行`relocate.py`的缺失文件检查确认是否完整。

复制后的JSON含有旧服务器路径及冻结SHA；直接改一个根目录会导致校验失败。**只在副本上**运行迁移工具，先预览：

```bash
python relocate.py --data-root workspace/pipeline_exp --old-workspace /old/ArtiFixer
# 修正缺失文件，确认映射后显式写入：
python relocate.py --data-root workspace/pipeline_exp --old-workspace /old/ArtiFixer --apply
```

工具从元数据识别NVS/Omega/Wan原路径，用上述环境变量指向新权重。额外路径用`--map '/old/prefix=/new/prefix'`，如自定义权重文件名或额外服务目录。缺失文件阻止写入；原JSON备份在`workspace/migration_backups/`。工具更新路径、权重文件元数据、冻结文件哈希及下游哈希引用。这是有记录的新迁移谱系，不把新哈希冒充原始冻结结果；原机实验应保持不变。

## 运行入口与回归

`run.py`设置工作目录和Python导入路径，避免脚本误用另一个已安装的运行时：

```bash
python run.py --list
python run.py vggtdiff_consistency_abc.py --help
python run.py vggtdiff_consistency_abc.py preflight
```

迁移已有数据后先`preflight`及`baseline`；准备新数据则先`prepare`、`segment --scope real`，实际检查掩码后用`review --scope real --note '实际检查记录'`记录，再运行基线。

当前A/B/C固定16个真实目标相机、三个种子、832×480、50步。A0=0～7；A1/B1=8～15；C=4～11；OA/OB=6～13。B1/OB使用A0前六张生成图，相机6、7不进入接力条件。先运行真实参考各组，再检查A0图像及掩码，最后运行接力组：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py vggtdiff_consistency_abc.py infer --calls A0 A1 C OA
CUDA_VISIBLE_DEVICES=0 python run.py vggtdiff_consistency_abc.py segment --scope generated --calls A0 A1 C OA
# 实际审阅各seed的contact.jpg/mask_review.jpg后：
python run.py vggtdiff_consistency_abc.py review --scope generated --calls A0 A1 C OA --note '实际检查记录'
CUDA_VISIBLE_DEVICES=0 python run.py vggtdiff_consistency_abc.py infer --calls B1 OB
CUDA_VISIBLE_DEVICES=0 python run.py vggtdiff_consistency_abc.py segment --scope generated --calls B1 OB
python run.py vggtdiff_consistency_abc.py review --scope generated --calls B1 OB --note '实际检查记录'
python run.py vggtdiff_consistency_abc.py evaluate
python run.py vggtdiff_consistency_abc.py report
```

`--seeds 20261026`可拆成单种子任务。**一个(seed,call)仅由一个进程写入，不要让多个worker同时生成同一输出目录。** 可用不同GPU分别运行不同seed。迁移后首次读取网络盘权重可能很慢；建议权重放在目标服务器本地高速盘。

CPU测试：

```bash
PYTHONPATH=workspace/scripts:workspace:workspace/worldact:workspace/thirdparty/VGGT-Diff \
  python -m pytest -q workspace/tests/test_crossview_geometry*.py \
  workspace/tests/test_vggtdiff_consistency_abc.py workspace/tests/test_vggtdiff_batch07_continuous.py \
  workspace/tests/test_vggtdiff_noise_control.py workspace/tests/test_car15*.py tests
PYTHONPATH=workspace/thirdparty/VGGT-Diff python -m pytest -q workspace/thirdparty/VGGT-Diff/tests
```

## 已有结果与未完成部分

当前三层评估的真实白车基线：29/29近邻图对匹配充分；极线误差中位数0.157px；三视角留出误差中位数/P90约0.598/2.555px。相机为估计值，不能当作标定位姿GT。

2026-10-23继续生成时GPU7发生启动失败，其余8个调用完成去噪但在VAE解码阶段同样失败。**这次尝试没有成功保存新生成图，也没有可恢复latent检查点**；不能标记A/B/C实验完成。历史第6/7/8批、3DGS、正则及诊断结果属于另外已保存的实验，不应与这次失败尝试混淆。设备恢复并迁移完成后需要重新生成。

主评估不先按极线误差筛匹配；白车与背景独立，低支持标证据不足，同相机比较不做配准。批内/批间兼容不代表全局三维真实；无真实覆盖区域只评价局部兼容性与漂移风险。
