# Frozen white-car E1 benchmark

`vggt-gen-v1` records the cameras, recipe, reference metrics and SHA-256 manifest of the original `stage2/E1` experiment. Full images, masks and initialization PLY remain local and are not included in the public branch. Supply the complete fixture directory with `--example /path/to/white_car_fixture`.
This is a reproducible baseline for the packaged generator and reconstructor.
The video preprocessing workflow is retained separately; re-estimating its cameras,
cloud and orbit creates a new experiment rather than this frozen benchmark.

## Reconstruct and validate

Use the reconstruction environment described in the repository README, with CUDA,
gsplat 1.4.0 and a 3DGRUT checkout. No diffusion weights or original ArtiFixer data
directories are needed for this command:

```bash
python scripts/reproduce_white_car_e1.py --example /path/to/white_car_fixture --verify-only

CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_white_car_e1.py --example /path/to/white_car_fixture \
  --threedgrut /path/to/3DGRUT \
  --output outputs/white_car_e1_v1
```

The entry point checks all files against `benchmark.json`, snapshots the fixture
inside the output directory, trains 79 views for 49,375 steps, evaluates the same
79 training cameras and 16 independent real cameras, and renders the original
160-camera evaluation orbit. Every training view receives exactly 625 updates.
An input mismatch or an evaluation outside the recorded tolerances exits with an
error. Use a new output directory after an interrupted training run; partial
training is not silently restarted or overwritten.

Outputs:

```text
outputs/white_car_e1_v1/
├── fixture/                       # Immutable inputs for this run
├── e1/model.ply
├── e1/training/result.json
├── evaluation/benchmark.json       # Actual/reference metrics and pass/fail
├── evaluation/orbit360.mp4         # Same independent 160-camera orbit as old E1
└── evaluation/orbit_contact.jpg
```

The reference medians are:

| Set | Foreground PSNR | Silhouette IoU | Outside opacity / mask area |
| --- | ---: | ---: | ---: |
| 79 training views | 24.3939 dB | 0.90585 | 0.10322 |
| 16 independent real views | 17.7400 dB | 0.88703 | 0.10885 |

The old model contains 77,270 Gaussians. Count is reported as a diagnostic rather
than requiring a bit-identical training result. Acceptance requires absolute
differences within 0.5 dB PSNR, 0.02 IoU and 0.03 outside opacity on both sets.
These are regression tolerances, not a claim that the unseen side matches the
physical car. Single-run results can vary with the CUDA/PyTorch/gsplat environment.

The actual `vggt-gen-v1` validation reproduced training PSNR 24.2480 dB / IoU
0.90367 and real-view PSNR 17.7498 dB / IoU 0.88629; all six checks passed.
See [the validation record](../../docs/white_car_e1_v1_validation.md). Regeneration
was checked separately: its median PSNR against the old PNGs is 49.0943 dB,
with small pixel differences rather than exact equality.

## Check the generator separately

Use the generation environment with the Trajectory checkpoint and VGGT-Omega:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_trajectory.py \
  --checkpoint /path/to/VGGT-Diff-Trajectory-480x832/vggt-diff.safetensors \
  --omega-checkpoint /path/to/vggt_omega_1b_512.pt \
  --base-model-dir /path/to/Wan2.1-I2V-14B-480P \
  --example /path/to/white_car_fixture \
  --output outputs/white_car_generation_v1 \
  --height 480 --width 832 --steps 50 --seed 20261030 --vram-limit-gib 46

python scripts/reproduce_white_car_e1.py --example /path/to/white_car_fixture --verify-only \
  --generated-frames outputs/white_car_generation_v1/frames \
  --output outputs/white_car_generation_v1
```

The second command compares all 80 decoded PNGs to the frozen run. It requires
identical pixels before reusing the old masks, and saves a per-frame report. If
they differ, reconstruct the frozen frames to reproduce E1, or segment the new
images into a separate mask directory for a new experiment. Do not mix old masks
with a new generated orbit. The generator's generic existence-based cache still
requires a fresh output directory whenever its inputs or settings change.

## What is fixed

- Six real source frames in order: 1481, 1240, 937, 635, 272, 2630.
- These are distortion-corrected, rotation-rectified virtual camera images,
  not bounding-box crops. Their source W2C/K are kept exactly as calibrated.
  The first five use fx 964.6449; frame 2630 uses fx 1363.0180.
- `trajectory.json`: the original source cameras and 80 target cameras, in OpenCV
  W2C convention at 832×480. Target 79 repeats camera 0 and is excluded from training.
- `frames/` and `masks/`: the original 80 generated images and segmentations.
- `initial_points.ply`: 17,112 observed surface points followed by 4,000 volume
  seeds. SHA-256: `20aa1c27d023c7804bb215782d68747007103c767e815d5a01f2044ff39bc19b`.
- `initialization.json`: explicit bounds, seed opacity 0.01 and scene extent
  0.7391271937596919; reconstruction does not rely on an implicit white-car fallback.
- `evaluation/cameras.json`: the original 160-camera orbit plus 16 independent
  real evaluation views, whose frame IDs are disjoint from the source frame IDs.
- `evaluation/reference_metrics.json`: the original per-view evaluation values.
- `benchmark.json`: relative file paths, SHA-256 checksums, reference medians and
  acceptance tolerances. Data paths do not reference the original machine.

Model weights, training checkpoints and generated videos are excluded from Git.
Only camera records, configuration, checksums and metrics are published. A fresh checkout needs the external full fixture before reconstruction; no download is bundled. The fixture must retain the relative paths listed in `benchmark.json`.
