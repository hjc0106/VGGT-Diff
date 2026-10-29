# 其他推理与训练入口

[返回主说明](../README.md)

这些入口保留在本分支中，白车 E1 的验收范围见[验证记录](white_car_e1_v1_validation.md)。各脚本的完整参数请运行 `python scripts/<script>.py --help`。

## 推理

The bundled example uses six source views, an 80-frame camera trajectory, and the released full-resolution trajectory checkpoint:

```bash
python scripts/infer.py \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --example examples/garden \
  --height 480 \
  --width 832 \
  --steps 50 \
  --seed 20263012 \
  --fps 24 \
  --output outputs/garden
```

Outputs are saved to `outputs/garden/`. The model predicts 80 frames once and keeps those raw PNGs in `frames/`; `vggtdiff.mp4` appends the same frames in reverse order, producing a 160-frame, 24 fps forward-then-backward video without additional inference. Remove `--height 480 --width 832` for the lower, faster default resolution. CPU offloading is enabled by default; use `--vram-limit-gib 64` to retain more weights on a high-memory GPU or `--vram-limit-gib -1` to disable offloading.

> [!NOTE]
> The bundled Garden inputs and 80-camera path are the same scene used by the project-page trajectory demo. That demo used the effective-31K checkpoint and seed `20263012`; the released 36K successor uses the same protocol and produces a comparable trajectory. A repository revision published before October 18, 2026 accidentally paired the 832 x 480 PNGs with intrinsics still expressed at 336 x 192, which causes severe drift. Pull the latest revision before comparing results. CPU offloading changes memory use and runtime, not camera geometry or expected image content.

### Pose-free inference from six RGB images

VGGT-Omega first estimates the six source cameras, then VGGT-Diff generates 80 novel views. Defaults are 832 × 480, 12 fps, and route `0 → 1 → 2 → 3 → 4 → 5`; name inputs in physical walking order. The route interpolates camera centers and orientations without scene-collision checks.

```bash
python scripts/infer_pose_free.py \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --source-dir /path/to/six_rgb_images \
  --output outputs/pose_free
```

No ground-truth poses or frames are read. Outputs are `prediction.mp4`, `camera_trajectory.mp4`, synchronized `prediction_with_trajectory.mp4`, 80 PNG frames, and `cameras.json` containing the estimated source cameras and target path used. Camera views use small antialiased frustums. Reorder the default path with `--route-order 0 2 1 3 4 5`.

For a custom 80-camera path, pass `--trajectory-json /path/to/targets.json` with exactly one of:

- `target_w2c`: 80 homogeneous 4 × 4 OpenCV world-to-camera matrices in the VGGT-Omega world saved in `cameras.json`.
- `target_c2w_relative_to_source0`: 80 homogeneous 4 × 4 camera-to-world matrices relative to source camera 0.

Optional `target_intrinsics` supplies 80 pixel-space 3 × 3 matrices; otherwise, the median of the six estimated source intrinsics is used. Keep custom paths collision-free and at a sensible distance from observed views.

### Custom scenes

Use this layout:

```text
my_scene/
├── source_views/
│   ├── 00.png
│   ├── 01.png
│   └── ...
└── trajectory.json
```

`trajectory.json` contains `source_w2c`, `target_w2c`, `source_intrinsics`, and `target_intrinsics`. Extrinsics are OpenCV world-to-camera matrices; intrinsics are 3 × 3 pixel-space matrices at source-image resolution. Add `intrinsics_image_size: [width, height]` to make the loader reject an accidental image/intrinsics resolution mismatch.

```bash
python scripts/infer.py \
  --checkpoint checkpoints/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --example /path/to/my_scene \
  --output outputs/my_scene
```

## 训练

Training scenes follow the folder-form DL3DV convention:

```text
dataset_root/
└── scene_id/
    ├── images_4/
    └── transforms.json
```

First cache the frozen VGGT-Omega source features:

```bash
python scripts/prepare_omega_cache.py \
  --dataset-root /path/to/dataset_root \
  --cache-root /path/to/omega_cache \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt
```

### Training recipes

Both recipes freeze the Wan2.1 VAE and VGGT-Omega while training the DiT and conditioning modules. Their latent layouts differ, so their checkpoints are separate model families.

| <small>Recipe</small> | <small>Targets</small> | <small>Latent layout</small> | <small>Plucker poses</small> |
| --- | --- | --- | --- |
| <small>**Paper NVS**</small> | <small>Variable `N`-view set</small> | <small>Independent source/target encoding; `6 + N` slots</small> | <small>1 per target slot</small> |
| <small>**Continuous trajectory**</small> | <small>80 ordered, contiguous frames</small> | <small>Independent sources; causal target VAE; `6 + 21` slots</small> | <small>4 ordered poses per compressed slot</small> |

#### Paper NVS model

This is the independent-view protocol used for the quantitative results in the [paper](https://arxiv.org/abs/2609.33253). Six sources and `N` targets produce `6 + N` one-to-one image, camera, Plucker, and geometry slots without temporal VAE compression.

```bash
accelerate launch scripts/train.py \
  --training-mode nvs \
  --dataset-root /path/to/dataset_root \
  --omega-cache /path/to/omega_cache \
  --target-frames 1 2 4 \
  --height 192 --width 336 \
  --backbone-lr 1e-5 \
  --condition-lr 1e-4 \
  --output outputs/nvs_training_run
```

The paper progresses the half-resolution target curriculum through `{1,2,4}` -> `{2,4,8}` -> `{4,8,12}` -> `{4,8,12,16}`, then trains at `480x832` with `{4,8}` -> `{4,8,12}` -> `{4,8,12,16}` and a `5e-6` backbone learning rate. Resume the previous stage and change `--target-frames`, resolution, and learning rate accordingly.

#### Continuous 80-frame trajectories

An 80-frame target is padded to 81 frames and causally compressed into 21 target slots. Camera conditions bypass the VAE; the four ordered Plucker maps in each causal window are packed and projected by the temporal Plucker adapter.

```bash
accelerate launch scripts/train.py \
  --training-mode trajectory \
  --dataset-root /path/to/dataset_root \
  --omega-cache /path/to/omega_cache \
  --resume /path/to/vggtdiff_trajectory.safetensors \
  --target-frames 80 \
  --height 480 \
  --width 832 \
  --output outputs/trajectory_training_run
```

The paper's original trajectory adaptation uses `192x336`; the command above selects the full-resolution continuation. Both recipes use source-anchor camera normalization and point-track residual consistency.
