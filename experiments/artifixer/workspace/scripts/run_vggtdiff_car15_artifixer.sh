#!/usr/bin/env bash
set -euo pipefail
cd "${ARTIFIXER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SCENE_ID=vggtdiff_mixed15_orbit64_front_views
export SCENE_FRONT=$SCENE_ID
export PIPELINE_ROOT=pipeline_exp/realworld/scene-1-nofull/06_worldact_white_car_v1/artifixer_episode_chain_64x4/vggtdiff_car15_reconstruction_v1/artifixer_2d_v1
export PREPARED_ROOT=$PIPELINE_ROOT/prepared
export PREPARED_SCENE=$PREPARED_ROOT/$SCENE_ID
export CAPTION_H5=$PREPARED_SCENE/captions/$SCENE_ID/caption_white_car.h5
export CAPTION_NAME=caption_white_car.h5
export SAVE_DIR=$PIPELINE_ROOT/results
export WORLDACT_PYTHON=${PYTHON:-python}
export ANALYSIS_GPU=2
export INFER_GPUS=2
export INFER_NPROC_PER_NODE=1
export CONTEXT_PARALLEL_SIZE=1
export INFERENCE_STEPS=4
export INFERENCE_SEED=0
export MASTER_PORT=29723
export CONTAINER_NAME=artifixer-vggtdiff-car15-v1
export HARD_PRESERVE_BACKGROUND=1
export COMPOSITE_OUTSIDE_ROI=1
./scripts/run_tank.sh "${1:-analyze}"
