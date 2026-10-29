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

COLMAP 使用系统里的命令行程序，特征提取限制长边 1600 并使用 CPU。顺序匹配的重叠是 15 帧，并用二次重叠补回环；回环检测需要词表，这里没有启用。`--frames` 里不在均匀采样中的帧会补进已有重建，不用重跑整段 SfM。参考图不再裁剪。第 3 步先用每张图掩码框中心的射线求一个车辆中心，再让每个 COLMAP 相机保持位置不动、无滚转地看向这个中心，用 `source_focal_px`（默认 965）和居中的主点重新采样，图像和 K/W2C 一起写进 `sources.json`，旋转量记在 `virtual_rotation_deg`。车放不下时只降低这一张的焦距；重采样视野超出原图时只提高这一张的焦距，不留黑边。第 4 步在这些图上跑 Omega，只取它的逐像素深度：按相机中心的相似变换换成 COLMAP 尺度，再沿已知相机的射线反投影，所以点云和源相机在同一个 COLMAP 坐标系里。`4_cameras/align.json` 记录 Omega 与这些相机在位姿、焦距上的差别，以及每个视角的 `projection_hit`；落在画面内的点至少要有 70% 落在该图的 SAM3 掩码上，否则这一步失败。点云包围盒中心和第 3 步的瞄准中心相差超过相机距离的 2%（`aim_refine_ratio`）时，`run` 会把点云中心写进 `3_sources/overrides.json` 的 `center`（`center_source` 为 `cloud`），重做第 3、4 步一次；手动写的 `center` 不会被改。第 5 步的轨道绕点云中心、沿 SfM 上方向旋转，半径和高度取源相机的中位数，起点在第 0 张参考图的方位，相机都看向中心；目标焦距取源焦距的中位数，主点在 (415.5, 239.5)。`--frames` 的顺序会保留，第 0 张参考图就是轨道起点。要沿用另一次实验的目标相机，在 `5_orbit/overrides.json` 里把 `reference_trajectory` 设为那份 `trajectory.json`（需要带 `source_frames`）；这一步按同一批视频帧的源相机中心求相似变换，把它的全部目标位姿搬进当前坐标系，残差写在 `orbit.json` 的 `reference_fit` 里。SAM3 默认置信度是 0.15。生成帧里没有合格掩码，或掩码面积和相邻帧差了 3 倍以上时，这一步失败并写出 `6_frame_masks/needs_review.jpg`；在 `overrides.json` 的 `candidates` 里指定候选编号后再跑。低分候选只保存在 `candidates_low/`，不会写进训练用的掩码。SAM3 和 VGGT-Omega 都在生成环境里。权重和已安装的 SAM3 包不匹配时，加上 `--sam3-root` 指向对应源码目录。Wan2.1 的分词器读 `google/umt5-xxl`。

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

从视频一路跑到 E1：

```bash
python scripts/run_traj_e1.py \
  --generate-python .venv-gen/bin/python \
  --reconstruct-python .venv-e1/bin/python \
  --video /path/to/video.mp4 \
  --prompt "white car" \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --sam3-checkpoint /path/to/sam3.1_multiplex.pt \
  --base-model-dir /path/to/Wan2.1-I2V-14B-480P \
  --output outputs/white_car \
  --threedgrut /path/to/3DGRUT
```

这条命令先做第 1 到第 5 步，再生成轨道，然后做第 6 步分割，最后重建。已经准备好的场景仍然用下面的命令。
