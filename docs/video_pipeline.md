# 从视频准备新场景

[返回主说明](../README.md)

这条流程会重新估计图像、相机、初始化和轨迹，属于新实验；白车旧 E1 的已验收复现使用仓库固定输入，见[复现说明](../examples/white_car/README.md)。


`gen_pro` 把一段绕物体拍摄的视频做成上面的场景。六个步骤各自写入工作目录，并可以用 `overrides.json` 改结果后从该步重跑：

| 步骤 | 目录 | 做什么 | 可以改 |
| --- | --- | --- | --- |
| 1 | `1_video` | 抽清晰帧、COLMAP、估计物体中心和上方向 | `center`、`up` |
| 2 | `2_select` | 按方位角选 6 张参考帧 | `frames` |
| 3 | `3_sources` | 去畸变、SAM3 分割，相机绕自身中心转向车辆中心，按固定焦距重采样成 832×480 | `prompt`、`candidates`、`center`、`source_focal_px` |
| 4 | `4_cameras` | 用 Omega 深度在源相机上生成点云；表面点必须投进源图掩码 | `confidence_quantile`、`outlier_drop`、`projection_min_hit` |
| 5 | `5_orbit` | 80 帧闭合轨道并导出 `scene/` | `radius_scale`、`height_offset`、`reference_trajectory` |
| 6 | `6_frame_masks` | 生成之后分割 80 帧 | `candidates` |

COLMAP 支持系统命令行程序，以及通过 `--colmap-python` 指定的独立 PyCOLMAP 环境。特征提取限制长边 1600；`--sfm-device auto` 根据所选安装包的 CUDA 支持选择提取和匹配设备，也可用 `cpu` 或 `cuda` 显式指定。显式请求 CUDA 而安装包不支持时会提前报错。顺序匹配的重叠是 15 帧，并用二次重叠补回环；回环检测需要词表，这里没有启用。`--frames` 里不在均匀采样中的帧会补进已有重建，不用重跑整段 SfM。参考图不再裁剪。第 3 步先用每张图掩码框中心的射线求一个车辆中心，再让每个 COLMAP 相机保持位置不动、无滚转地看向这个中心，用 `source_focal_px`（默认 965）和居中的主点重新采样，图像和 K/W2C 一起写进 `sources.json`，旋转量记在 `virtual_rotation_deg`。车放不下时只降低这一张的焦距；重采样视野超出原图时只提高这一张的焦距，不留黑边。第 4 步在这些图上跑 Omega，只取它的逐像素深度：按相机中心的相似变换换成 COLMAP 尺度，再沿已知相机的射线反投影，所以点云和源相机在同一个 COLMAP 坐标系里。`4_cameras/align.json` 记录 Omega 与这些相机在位姿、焦距上的差别，以及每个视角的 `projection_hit`；落在画面内的点至少要有 70% 落在该图的 SAM3 掩码上，否则这一步失败。点云包围盒中心和第 3 步的瞄准中心相差超过相机距离的 2%（`aim_refine_ratio`）时，`run` 会把点云中心写进 `3_sources/overrides.json` 的 `center`（`center_source` 为 `cloud`），重做第 3、4 步一次；手动写的 `center` 不会被改。第 5 步的轨道绕点云中心、沿 SfM 上方向旋转，半径和高度取源相机的中位数，起点在第 0 张参考图的方位，相机都看向中心；目标焦距取源焦距的中位数，主点在 (415.5, 239.5)。`--frames` 的顺序会保留，第 0 张参考图就是轨道起点。要沿用另一次实验的目标相机，在 `5_orbit/overrides.json` 里把 `reference_trajectory` 设为那份 `trajectory.json`（需要带 `source_frames`）；这一步按同一批视频帧的源相机中心求相似变换，把它的全部目标位姿搬进当前坐标系，残差写在 `orbit.json` 的 `reference_fit` 里。SAM3 默认置信度是 0.15。生成帧里没有合格掩码，或掩码面积和相邻帧差了 3 倍以上时，这一步失败并写出 `6_frame_masks/needs_review.jpg`；在 `overrides.json` 的 `candidates` 里指定候选编号后再跑。低分候选只保存在 `candidates_low/`，不会写进训练用的掩码。SAM3 和 VGGT-Omega 都在生成环境里。权重和已安装的 SAM3 包不匹配时，加上 `--sam3-root` 指向对应源码目录。Wan2.1 的分词器读 `google/umt5-xxl`。

```bash
python -m gen_pro run \
  --work work/white_car \
  --video /path/to/video.mp4 \
  --prompt "white car" \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --sam3-checkpoint /path/to/sam3.1_multiplex.pt \
  --until select

python -m gen_pro run --work work/white_car --from sources --until orbit
python -m gen_pro status --work work/white_car
```

`--until select` 停在选帧之后，方便改 `work/white_car/2_select/overrides.json` 里的 `frames`。指定视频帧号也可以直接传 `--frames 1481 1240 937 635 272 2630`。工作目录不进 git。

### GPU COLMAP

系统包可能没有编译 CUDA；只有 NVIDIA GPU 和 PyTorch CUDA 并不能让它启用 GPU。可按 [COLMAP 官方说明](https://colmap.github.io/pycolmap/index.html) 安装官方 `pycolmap-cuda12`，使用独立环境避免影响生成环境的依赖：

```bash
python -m venv work/tools/colmap-gpu
work/tools/colmap-gpu/bin/python -m pip install pycolmap-cuda12==4.2.1

CUDA_VISIBLE_DEVICES=0 python -m gen_pro run \
  --work work/scene_gpu \
  --video /path/to/video.mp4 \
  --colmap-python "$PWD/work/tools/colmap-gpu/bin/python" \
  --sfm-device cuda --sfm-gpu-index 0 --sfm-num-threads 16 \
  --until video
```

`--sfm-gpu-index` 使用 `CUDA_VISIBLE_DEVICES` 之后的设备编号。上述参数会保存在工作目录的 `config.json`，后续选帧及补注册沿用同一个后端。GPU 路径也使用 SIFT、顺序匹配、共享 OPENCV 相机和增量 SfM；补注册固定已有相机位姿和内参。

GPU 特征提取和匹配不代表整个建图过程都在 GPU 上。默认 `--sfm-ba-backend ceres` 使用 CPU BA。官方 4.2.1 CUDA wheel 没有编译 Ceres CUDA/cuDSS 或 Caspar。进一步加速 BA 应优先评估支持 CUDA/cuDSS 的 Ceres 构建。Caspar 当前只支持 SIMPLE_RADIAL 和 PINHOLE，相机会使用其他模型时其观测可能被跳过；本流程使用 OPENCV，因此即使安装包编译了 Caspar，也会在开始重建前拒绝该选项。相关限制见 [COLMAP BA 加速说明](https://colmap.github.io/faq.html#speedup-bundle-adjustment)。

`1_video/colmap.log` 记录实际后端和设备；`colmap_timings.json` 记录 GPU 路径中提取、匹配、建图和导出的分段耗时，补注册单独写 `colmap_register_timings.json`。所有外部命令的耗时与退出码写在 `colmap_commands.json`；抽帧耗时写入该步骤的 metrics。抽帧目前仍使用 CPU 视频解码。

### SfM 速度配置

通过 `--sfm-profile` 选择增量 SfM 的 BA 调度，默认保留 `reference`。这些配置需要 `--colmap-python`，相机模型、SIFT 长边和顺序匹配重叠保持一致。

GPU 提取/匹配的两版用同一个开关控制：`--sfm-profile reference` 或 `--sfm-profile balanced`。开关不改变提取设备或 BA 设备；使用 `--sfm-device cuda --sfm-ba-device cpu` 固定这两项。完整入口 `scripts/run_traj_reconstruction.py` 接受并转发相同参数。

| 配置 | 建图帧 | 全局 BA 触发增长比例 | 每次最大细化轮数 | 建图后 BA |
| --- | --- | --- | --- | --- |
| `reference` | 全部候选帧 | 1.1 | 5 | 无额外 BA |
| `balanced` | 全部候选帧 | 1.3 | 2 | 最多 100 次迭代 |
| `keyframe` | 默认 60 帧 | 1.3 | 2 | 最多 100 次迭代 |

局部 BA 和中间全局 BA 的最大迭代数仍分别为 25 和 50。`--sfm-final-ba-iterations` 可覆盖末尾 BA 次数，0 表示关闭。`balanced` 减少建图中重复的全局优化，再用末尾 BA 收敛；建议先使用它验证新场景。

```bash
# 只改这一项，即可在两版之间切换；比较实验建议使用不同工作目录
SFM_PROFILE=balanced  # 或 reference
CUDA_VISIBLE_DEVICES=0 python -m gen_pro run \
  --work "work/scene_${SFM_PROFILE}" --video /path/to/video.mp4 \
  --colmap-python "$PWD/work/tools/colmap-gpu/bin/python" \
  --sfm-device cuda --sfm-ba-device cpu --sfm-num-threads 16 \
  --sfm-profile "$SFM_PROFILE" \
  --until video
```

参数保存到 `config.json`，后续省略开关会沿用已保存的选择。对同一工作目录切换时，第 1 步会重新执行 SfM，并在完成后将下游步骤标为过期；完整流水线在相机轨迹变化时拒绝复用已有生成结果，需要新的 `--output` 目录。`--sfm-final-ba-iterations` 等显式高级配置仍会沿用，可在配置文件中将该项设回 `null` 恢复各 profile 的默认值。

`keyframe` 是实验选项。它保留指定参考帧和视频两端，再按时间覆盖选择其余帧。默认要求所选帧至少注册 95%，否则按 60 → 90 → 全部候选帧重试；全帧回退恢复 `reference` 的中间 BA 调度。可用 `--sfm-keyframes` 和 `--sfm-keyframe-min-register-ratio` 调整。未参与建图的帧仍保留在候选列表，选为参考图时按需补注册，已有相机固定。注册率通过并不能保证相机质量：白车 60/90 帧实验均未通过相机一致性检查，tank 的多次回退反而增加耗时，因此不建议默认启用。

`--sfm-ba-device cuda` 独立控制 Ceres BA，需使用实际编译了 CUDA 的 Ceres/PyCOLMAP 环境。启动前会检查求解器能力，官方 CUDA wheel 仅有 CUDA SIFT 时会拒绝此选项。局部小规模 BA 仍由 COLMAP 自行选择 CPU；这个参数不意味着完整 SfM 都运行在 GPU 上。当前机器额外构建了 COLMAP 4.2.1 + Ceres 2.2.0 + CUDA 12.8，入口为 `work/tools/colmap-cuda-ba/bin/python`；未启用 cuDSS，不能将其称为 cuDSS 稀疏 GPU 求解器。生成环境和系统 COLMAP 没有替换。

```bash
work/tools/colmap-cuda-ba/bin/python gen_pro/colmap_gpu.py --probe
# has_cuda 和 has_ceres_cuda 都应为 true
CUDA_VISIBLE_DEVICES=0 python -m gen_pro run \
  --work work/scene_gpu_ba --video /path/to/video.mp4 \
  --colmap-python "$PWD/work/tools/colmap-cuda-ba/bin/python" \
  --sfm-device cuda --sfm-profile balanced --sfm-ba-device cuda \
  --until video
```

2026-10-30 对固定候选帧的实验中，GPU SIFT + CPU BA 的 `reference` 与 `balanced` 分别耗时：白车 332.1 / 228.5 秒，tank 228.5 / 195.5 秒。注册相机数分别保持 125 和 99，指定六张参考图全部保留。按相似变换对齐同版本 `reference` 后，balanced 的旋转差 P95 均小于 0.04°、相机中心差 P95 均小于相机分布尺度的 0.08%。这是与基线的一致性检查，不是真值精度；耗时包括提取、匹配、建图和末尾 BA，不包含复用前的视频抽帧。详细记录位于工作目录 `work/sfm_optimization/20261030/`。

同一源码构建的 balanced CPU/GPU BA 对照中，白车分别为 230.5 / 248.2 秒，tank 为 208.3 / 193.6 秒，均通过相机一致性检查。实际求解摘要确认启用了 CUDA DENSE_SCHUR，但端到端收益并不稳定，当前推荐 CPU BA。以上是单次运行数据。

另从白车原始视频重新执行完整第一步，抽帧 48.0 秒、GPU SIFT + balanced CPU BA 221.4 秒，总计 269.8 秒，125/125 个相机注册；新抽出的 125 张 JPEG 与固定帧实验逐字节相同。这是当前配置的一次完整实测，未与旧 CPU 全流程做同条件计时对比。

包含同版本 CPU 提取/匹配对照的分段数据、关键帧失败结果及相机差异见 [v2 验证记录](sfm_v2_validation.md)。

从视频一路跑到 E1：

```bash
python scripts/run_traj_reconstruction.py \
  --generate-python .venv-gen/bin/python \
  --reconstruct-python .venv-e1/bin/python \
  --video /path/to/video.mp4 \
  --prompt "white car" \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --sam3-checkpoint /path/to/sam3.1_multiplex.pt \
  --base-model-dir /path/to/Wan2.1-I2V-14B-480P \
  --colmap-python "$PWD/work/tools/colmap-gpu/bin/python" \
  --sfm-device cuda --sfm-profile balanced --pose-refine \
  --output outputs/white_car \
  --threedgrut /path/to/3DGRUT
```

这条命令先做第 1 到第 5 步，再生成轨道，然后做第 6 步分割，最后重建。`--pose-refine` 可选，不加时保留固定相机的重建配方；也可独立使用 `scripts/reconstruction.py --pose-refine`。历史输出子目录仍叫 `e1`。
