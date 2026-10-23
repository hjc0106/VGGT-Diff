#!/usr/bin/env bash
# Run the tan-tank ArtiFixer preparation, analysis, and fixed V6-A inference workflow.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
# shellcheck source=run_art_common.sh
source "${REPO_ROOT}/run_art_common.sh"

log() {
  printf '[tank-v6a] %s\n' "$*"
}

has_npy() {
  local directory=$1
  compgen -G "${directory}/*.npy" >/dev/null
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_tank.sh [stage]

Stages:
  infer             Run fixed V6-A inference using existing analysis (default)
  analyze           Generate full-tank ROI, direction consistency, and disorder maps
  prepare           Run prepare_package, prepare_scene, regenerate, install_caption
  prepare_package   Prepare the WorldAct object package only
  prepare_scene     Prepare the ArtiFixer reconstruction scene only
  regenerate        Regenerate the front-view camera trajectory only
  install_caption   Install the existing strict tank caption only
  distill_check     Validate V6-A pred, anchors, and object_local.ply
  distill_smoke     D0: 1k-step MCMC distillation of raw pred
  distill           D1/GS distillation (see DISTILL_STRATEGY / DISTILL_RECON_ETA)
  distill_eval      Compare initial / pred / distilled renders and geometry
  distill_mv        Multi-view support of missing∩ROI pred repairs
  distill_import    Geometry guard + world-coordinate refined_world.ply
  all               Run prepare, analyze, then V6-A inference
  check             Validate inputs and print output locations

Useful overrides:
  INFER_GPUS=4,5,6,7 ANALYSIS_GPU=4 DISTILL_GPU=4
  FORCE_ANALYSIS=1                 Recompute existing analysis outputs
  SAVE_DIR=...                     Select a new inference output directory
  DISTILL_STEPS=5000 DISTILL_MODE=raw|region DISTILL_STRATEGY=mcmc|gs
  DISTILL_RECON_ETA=0|0.10|0.25    Pred-view L1/SSIM strength; 0 keeps LPIPS-only
  DISTILL_PRED_DIR=...             Must point at frames/batch_0000/pred, not pred_composited
  DISTILL_ALLOW_30K=1              Required before 30k steps
  DISTILL_MV_STRIDE=2              Pixel stride for distill_mv (raise to 4 if slow)
  FORCE_DISTILL=1                  Rebuild region GT / replace existing 3DGRUT outputs
  DOCKER_IMAGE=artifixer:cuda13-gsplat
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_STAGE=${1:-infer}

# Tank experiment layout. Every value remains overrideable from the shell.
export OBJECT_ID=${OBJECT_ID:-0}
export SCENE_ID=${SCENE_ID:-tan_tank_24}
export SCENE_FRONT=${SCENE_FRONT:-tan_tank_24_front_views}

export EXP_ROOT=${EXP_ROOT:-worldact_exp/tank_pipeline_plus}
export PIPELINE_ROOT=${PIPELINE_ROOT:-${EXP_ROOT}/artifixer_front_views_tank24}
export PREPARED_ROOT=${PREPARED_ROOT:-${PIPELINE_ROOT}/prepared}
export PREPARED_SCENE=${PREPARED_SCENE:-${PREPARED_ROOT}/${SCENE_FRONT}}

export STAGE_DIR=${STAGE_DIR:-${EXP_ROOT}/stage1}
export PLY=${PLY:-${EXP_ROOT}/stage2/object_0_foreground_tan_battle_tank_marked_24/object_0_foreground_tan_battle_tank_marked_24_final.ply}
export PACKAGE_OUT=${PACKAGE_OUT:-${PIPELINE_ROOT}/package}
export PACKAGE_DIR=${PACKAGE_DIR:-${PACKAGE_OUT}/object_${OBJECT_ID}/object_package}

export CAPTION_MODE=${CAPTION_MODE:-text}
export CAPTION_H5=${CAPTION_H5:-${PREPARED_ROOT}/_captions/${SCENE_ID}/caption_tank_strict.h5}
export CAPTION_NAME=${CAPTION_NAME:-caption_tank_strict.h5}
export CONFIG=${CONFIG:-worldact/configs/artifixer/car_front_views.yaml}

export DOCKER_IMAGE=${DOCKER_IMAGE:-artifixer:cuda13-gsplat}
export CAPTION_GPU=${CAPTION_GPU:-4}
export PREPARE_GPU=${PREPARE_GPU:-4}
export REGEN_GPU=${REGEN_GPU:-4}
export INFER_GPUS=${INFER_GPUS:-4,5,6,7}
export ANALYSIS_GPU=${ANALYSIS_GPU:-4}
export DISTILL_GPU=${DISTILL_GPU:-${ANALYSIS_GPU}}
export CACHE_HOST=${CACHE_HOST:-/tmp/artifixer-cache}
export FORCE_DISTILL=${FORCE_DISTILL:-0}
export DISTILL_MODE=${DISTILL_MODE:-raw}
export DISTILL_STRATEGY=${DISTILL_STRATEGY:-mcmc}
export DISTILL_STEPS=${DISTILL_STEPS:-5000}
export DISTILL_RECON_ETA=${DISTILL_RECON_ETA:-0}
export DISTILL_ALLOW_30K=${DISTILL_ALLOW_30K:-0}
export DISTILL_RING_BETA=${DISTILL_RING_BETA:-0.4}
INFERENCE_TRAJECTORY_DIR=artifixer-14b/distilled_views_reconstructed_colmap_auto12_evenly_spaced_sink7_trajectory
export DISTILL_PRED_DIR=${DISTILL_PRED_DIR:-${PIPELINE_ROOT}/inference_texture_disorder_v6_spill_a/${INFERENCE_TRAJECTORY_DIR}/${SCENE_FRONT}/frames/batch_0000/pred}

export RECON_ROOT=${RECON_ROOT:-${PREPARED_SCENE}/recon_results/${SCENE_FRONT}/reconstruction/${SCENE_FRONT}/ours_0}
export RENDER_DIR=${RENDER_DIR:-${RECON_ROOT}/renders}
export OPACITY_DIR=${OPACITY_DIR:-${RECON_ROOT}/opacity}
export FULL_TANK_ROI=${FULL_TANK_ROI:-${PREPARED_SCENE}/full_tank_roi}
export DIRECTION_ROOT=${DIRECTION_ROOT:-${PREPARED_SCENE}/gaussian_direction_consistency_v1_full_tank}
export TEXTURE_ANALYSIS=${TEXTURE_ANALYSIS:-${PREPARED_SCENE}/texture_disorder_v6a_inputs}
export LEGACY_TEXTURE_SCORE_ROOT=${LEGACY_TEXTURE_SCORE_ROOT:-${PREPARED_SCENE}/texture_disorder_v4_scores}
export LEGACY_PRIORITY_ROOT=${LEGACY_PRIORITY_ROOT:-${PREPARED_SCENE}/texture_disorder_v1_full_tank}

WORLDACT_PYTHON=${WORLDACT_PYTHON:-python}
FORCE_ANALYSIS=${FORCE_ANALYSIS:-0}
ANALYSIS_PYTHONPATH=${ANALYSIS_PYTHONPATH:-${REPO_ROOT}:${REPO_ROOT}/worldact:${REPO_ROOT}/thirdparty/3DGRUT-ArtiFixer}

run_front_stage() {
  local stage=$1
  log "front-view stage: ${stage}"
  STAGE="${stage}" ./scripts/run_artifixer_front_views_pipeline.sh
}

prepare_all() {
  [[ -f "${PLY}" ]] || { echo "Missing tank PLY: ${PLY}" >&2; exit 1; }
  [[ -f "${CAPTION_H5}" ]] || { echo "Missing strict tank caption: ${CAPTION_H5}" >&2; exit 1; }
  run_front_stage prepare_package
  run_front_stage prepare_scene
  run_front_stage regenerate
  run_front_stage install_caption
}

replace_arg=()
if [[ "${FORCE_ANALYSIS}" == "1" ]]; then
  replace_arg=(--replace)
fi

generate_roi() {
  if has_npy "${FULL_TANK_ROI}" && [[ "${FORCE_ANALYSIS}" != "1" ]]; then
    log "reuse full-tank ROI: ${FULL_TANK_ROI}"
    return
  fi
  [[ -d "${OPACITY_DIR}" ]] || { echo "Missing opacity directory: ${OPACITY_DIR}" >&2; exit 1; }
  [[ -d "${RENDER_DIR}" ]] || { echo "Missing render directory: ${RENDER_DIR}" >&2; exit 1; }
  log "generate full-tank ROI"
  "${WORLDACT_PYTHON}" scripts/generate_artifixer_full_tank_roi.py \
    --opacity-dir "${OPACITY_DIR}" \
    --render-dir "${RENDER_DIR}" \
    --output-dir "${FULL_TANK_ROI}" \
    --opacity-threshold 0.01 \
    --close-radius 8 \
    --dilate-radius 5 \
    --edge-blur-radius 2.0 \
    "${replace_arg[@]}"
}

generate_direction_metrics() {
  if [[ -f "${DIRECTION_ROOT}/gaussian_orientation_disorder.npy" && "${FORCE_ANALYSIS}" != "1" ]]; then
    log "reuse direction consistency: ${DIRECTION_ROOT}"
    return
  fi
  log "calculate Gaussian direction consistency"
  CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU}" \
  PYTHONPATH="${ANALYSIS_PYTHONPATH}" \
  "${WORLDACT_PYTHON}" scripts/visualize_gaussian_direction_consistency.py \
    --prepared-dir "${PREPARED_SCENE}" \
    --roi-dir "${FULL_TANK_ROI}" \
    --output-dir "${DIRECTION_ROOT}" \
    --device cuda \
    --neighbors 16 \
    --minimum-planarity 0.15 \
    "${replace_arg[@]}"
}

generate_texture_disorder() {
  if has_npy "${TEXTURE_ANALYSIS}/region_low_precision" && \
     has_npy "${TEXTURE_ANALYSIS}/region_missing" && \
     has_npy "${TEXTURE_ANALYSIS}/diffusion_edit_priority" && \
     [[ "${FORCE_ANALYSIS}" != "1" ]]; then
    log "reuse texture-disorder analysis: ${TEXTURE_ANALYSIS}"
    return
  fi
  log "calculate texture-disorder maps"
  CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU}" \
  PYTHONPATH="${ANALYSIS_PYTHONPATH}" \
  "${WORLDACT_PYTHON}" scripts/visualize_artifixer_texture_disorder.py \
    --prepared-dir "${PREPARED_SCENE}" \
    --roi-dir "${FULL_TANK_ROI}" \
    --direction-dir "${DIRECTION_ROOT}" \
    --output-dir "${TEXTURE_ANALYSIS}" \
    --device cuda \
    --missing-alpha-low 0.03 \
    --missing-alpha-high 0.35 \
    --disorder-low 0.42 \
    --disorder-high 0.68 \
    "${replace_arg[@]}"
}

analyze_all() {
  [[ -f "${PREPARED_SCENE}/split.json" ]] || { echo "Missing prepared scene: ${PREPARED_SCENE}/split.json" >&2; exit 1; }
  [[ -x "${WORLDACT_PYTHON}" ]] || { echo "Python is not executable: ${WORLDACT_PYTHON}" >&2; exit 1; }
  generate_roi
  generate_direction_metrics
  generate_texture_disorder
}

select_v6_map_dirs() {
  if [[ -z "${TEXTURE_DISORDER_DIR:-}" ]]; then
    if has_npy "${TEXTURE_ANALYSIS}/region_low_precision"; then
      TEXTURE_DISORDER_DIR=${TEXTURE_ANALYSIS}/region_low_precision
    else
      TEXTURE_DISORDER_DIR=${LEGACY_TEXTURE_SCORE_ROOT}/region_low_precision
    fi
  fi
  if [[ -z "${REGION_MISSING_DIR:-}" ]]; then
    if has_npy "${TEXTURE_ANALYSIS}/region_missing"; then
      REGION_MISSING_DIR=${TEXTURE_ANALYSIS}/region_missing
    else
      REGION_MISSING_DIR=${LEGACY_TEXTURE_SCORE_ROOT}/region_missing
    fi
  fi
  if [[ -z "${REPAIR_PRIORITY_DIR:-}" ]]; then
    if has_npy "${TEXTURE_ANALYSIS}/diffusion_edit_priority"; then
      REPAIR_PRIORITY_DIR=${TEXTURE_ANALYSIS}/diffusion_edit_priority
    else
      REPAIR_PRIORITY_DIR=${LEGACY_PRIORITY_ROOT}/diffusion_edit_priority
    fi
  fi
  export TEXTURE_DISORDER_DIR REGION_MISSING_DIR REPAIR_PRIORITY_DIR
}

validate_v6_inputs() {
  select_v6_map_dirs
  local missing=0
  local required=(
    "${PREPARED_SCENE}/split.json"
    "${CAPTION_H5}"
  )
  local path
  for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required input: ${path}" >&2
      missing=1
    fi
  done
  local map_dir
  for map_dir in "${TEXTURE_DISORDER_DIR}" "${REGION_MISSING_DIR}" "${REPAIR_PRIORITY_DIR}"; do
    if ! has_npy "${map_dir}"; then
      echo "Missing NPY maps: ${map_dir}" >&2
      missing=1
    fi
  done
  [[ "${missing}" == "0" ]] || exit 1
}

run_v6a_inference() {
  validate_v6_inputs

  export LATENT_PRESERVE_MODE=texture_disorder_train_aligned
  export TEXTURE_DISORDER_DIR REGION_MISSING_DIR REPAIR_PRIORITY_DIR

  export DISORDER_EXTRA_NOISE_SCALE=${DISORDER_EXTRA_NOISE_SCALE:-0.15}
  export DISORDER_EXTRA_NOISE_CAP=${DISORDER_EXTRA_NOISE_CAP:-0.15}
  export DISORDER_SUPPORT_THRESHOLD=${DISORDER_SUPPORT_THRESHOLD:-0.10}
  export DISORDER_SUPPORT_GATE_START=${DISORDER_SUPPORT_GATE_START:-0.10}
  export DISORDER_SUPPORT_GATE_FULL=${DISORDER_SUPPORT_GATE_FULL:-0.35}
  export DISORDER_GAMMA=${DISORDER_GAMMA:-1.5}
  export DISORDER_SOLID_OPACITY=${DISORDER_SOLID_OPACITY:-0.5}
  export DISORDER_HIGH_PRECISION_OPACITY=${DISORDER_HIGH_PRECISION_OPACITY:-0.8}
  export DISORDER_PRESERVE_THRESHOLD=${DISORDER_PRESERVE_THRESHOLD:-0.02}

  # Fixed V6-A profile. V7 zoom and timestep schedules remain disabled.
  export DISORDER_SPILL_CAP=${DISORDER_SPILL_CAP:-0.03}
  export DISORDER_SOFT_PRESERVE=${DISORDER_SOFT_PRESERVE:-0}
  export DISORDER_GATE_HARD_PRESERVE=${DISORDER_GATE_HARD_PRESERVE:-0}
  export ROI_ZOOM=0
  export DISORDER_CORE_PRESERVE_SCHEDULE=
  export DISORDER_RING_PRESERVE_SCHEDULE=

  export REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH=${REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH:-0.80}
  export REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH=${REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH:-1.00}
  export HARD_PRESERVE_BACKGROUND=${HARD_PRESERVE_BACKGROUND:-1}
  export COMPOSITE_OUTSIDE_ROI=${COMPOSITE_OUTSIDE_ROI:-1}
  export SAVE_LATENT_MASK_DIAGNOSTICS=${SAVE_LATENT_MASK_DIAGNOSTICS:-1}
  export INFERENCE_SEED=${INFERENCE_SEED:-0}
  export SAVE_DIR=${SAVE_DIR:-${PIPELINE_ROOT}/inference_texture_disorder_v6a_fixed}

  log "run V6-A inference -> ${SAVE_DIR}"
  run_front_stage infer
}

print_status() {
  validate_v6_inputs
  local save_dir=${SAVE_DIR:-${PIPELINE_ROOT}/inference_texture_disorder_v6a_fixed}
  local result_root=${save_dir}/artifixer-14b/distilled_views_reconstructed_colmap_auto12_evenly_spaced_sink7_trajectory/${SCENE_FRONT}
  log "inputs OK"
  printf 'prepared scene:      %s\n' "${PREPARED_SCENE}"
  printf 'texture analysis:    %s\n' "${TEXTURE_ANALYSIS}"
  printf 'low-precision maps:  %s\n' "${TEXTURE_DISORDER_DIR}"
  printf 'missing maps:        %s\n' "${REGION_MISSING_DIR}"
  printf 'priority maps:       %s\n' "${REPAIR_PRIORITY_DIR}"
  printf 'pred:                %s\n' "${result_root}/frames/batch_0000/pred"
  printf 'pred_composited:     %s\n' "${result_root}/frames/batch_0000/pred_composited"
  printf 'mask diagnostics:    %s\n' "${result_root}/mask_diagnostics"
  printf 'distill pred:        %s\n' "${DISTILL_PRED_DIR}"
  printf 'distill output:      %s\n' "$(distill_output_root)"
}

distill_frame_indices() {
  local joined
  joined=$(printf '%s,' {0..31})
  printf '%s\n' "${joined%,}"
}

distill_steps_tag() {
  if (( DISTILL_STEPS % 1000 == 0 )); then
    printf '%sk\n' "$((DISTILL_STEPS / 1000))"
  else
    printf '%s\n' "${DISTILL_STEPS}"
  fi
}

distill_recon_tag() {
  case "${DISTILL_RECON_ETA}" in
    0|0.0|0.00) printf '' ;;
    0.1|0.10) printf '_r10' ;;
    0.25) printf '_r25' ;;
    *) printf '_r%s' "${DISTILL_RECON_ETA}" ;;
  esac
}

distill_output_root() {
  local tag recon
  tag=$(distill_steps_tag)
  recon=$(distill_recon_tag)
  if [[ "${DISTILL_MODE}" == "region" ]]; then
    printf '%s\n' "${PREPARED_SCENE}/artifixer3d_v6a_region${recon}_${tag}"
  elif [[ "${DISTILL_STRATEGY}" == "gs" ]]; then
    printf '%s\n' "${PREPARED_SCENE}/artifixer3d_v6a_pred_gs${recon}_${tag}"
  else
    printf '%s\n' "${PREPARED_SCENE}/artifixer3d_v6a_pred${recon}_${tag}"
  fi
}

distill_frames_dir() {
  if [[ "${DISTILL_MODE}" == "region" ]]; then
    printf '%s\n' "${PREPARED_SCENE}/artifixer3d_v6a_region_supervision/pred"
  else
    printf '%s\n' "${DISTILL_PRED_DIR}"
  fi
}

distill_config_name() {
  if [[ "${DISTILL_MODE}" == "region" || "${DISTILL_STRATEGY}" == "gs" ]]; then
    printf '%s\n' "apps/colmap_3dgut_sparse_lpips"
  else
    printf '%s\n' "apps/colmap_3dgut_sparse_mcmc_lpips"
  fi
}

distill_render_dir() {
  printf '%s\n' "$(distill_output_root)/recon_results/${SCENE_FRONT}/artifixer3d/${SCENE_FRONT}/ours_${DISTILL_STEPS}/renders"
}

run_distill_python() {
  PYTHONPATH="${ANALYSIS_PYTHONPATH}" "${WORLDACT_PYTHON}" -m data_processing.artifixer3d_distill "$@"
}

assert_distill_limits() {
  if [[ "${DISTILL_STEPS}" -ge 30000 && "${DISTILL_ALLOW_30K}" != "1" ]]; then
    echo "30k distillation is deferred; set DISTILL_ALLOW_30K=1 to override." >&2
    exit 1
  fi
  if [[ "${DISTILL_PRED_DIR}" == */pred_composited ]]; then
    echo "DISTILL_PRED_DIR must be raw pred, not pred_composited: ${DISTILL_PRED_DIR}" >&2
    exit 1
  fi
  if [[ "${DISTILL_MODE}" != "raw" && "${DISTILL_MODE}" != "region" ]]; then
    echo "DISTILL_MODE must be raw or region, got ${DISTILL_MODE}" >&2
    exit 1
  fi
  if [[ "${DISTILL_STRATEGY}" != "mcmc" && "${DISTILL_STRATEGY}" != "gs" ]]; then
    echo "DISTILL_STRATEGY must be mcmc or gs, got ${DISTILL_STRATEGY}" >&2
    exit 1
  fi
  if awk "BEGIN {exit !(${DISTILL_RECON_ETA} >= 1)}"; then
    echo "DISTILL_RECON_ETA>=1 is deferred; use 0, 0.10, or 0.25." >&2
    exit 1
  fi
}

run_distill_check() {
  assert_distill_limits
  log "distill_check pred=${DISTILL_PRED_DIR}"
  run_distill_python check \
    --scene_root "${PREPARED_SCENE}" \
    --pred_dir "${DISTILL_PRED_DIR}" \
    --scene_id "${SCENE_FRONT}"
}

prepare_region_supervision_if_needed() {
  if [[ "${DISTILL_MODE}" != "region" ]]; then
    return
  fi
  select_v6_map_dirs
  local output_dir=${PREPARED_SCENE}/artifixer3d_v6a_region_supervision
  if [[ -f "${output_dir}/pred/00000.png" && "${FORCE_DISTILL}" != "1" ]]; then
    log "reuse region supervision: ${output_dir}/pred"
    return
  fi
  log "build region-protected distill GT -> ${output_dir}"
  run_distill_python prepare-region \
    --pred_dir "${DISTILL_PRED_DIR}" \
    --render_dir "${RENDER_DIR}" \
    --output_dir "${output_dir}" \
    --opacity_dir "${OPACITY_DIR}" \
    --disorder_dir "${TEXTURE_DISORDER_DIR}" \
    --roi_dir "${FULL_TANK_ROI}" \
    --missing_dir "${REGION_MISSING_DIR}" \
    --frame_indices "$(distill_frame_indices)" \
    --ring_beta "${DISTILL_RING_BETA}"
}

run_distill() {
  local steps=${1:-${DISTILL_STEPS}}
  DISTILL_STEPS=${steps}
  assert_distill_limits
  run_distill_check
  prepare_region_supervision_if_needed

  local out_root frames_dir config_name container_name
  out_root=$(distill_output_root)
  frames_dir=$(distill_frames_dir)
  config_name=$(distill_config_name)
  container_name="artifixer3d_tank_${DISTILL_MODE}_${DISTILL_STRATEGY}$(distill_recon_tag)_$(distill_steps_tag)"
  mkdir -p "${out_root}"

  local cmd=(
    python -m data_processing.run_artifixer3d
    --scene_root "${PREPARED_SCENE}"
    --artifixer_frames_dir "${frames_dir}"
    --initial_ply "${PREPARED_SCENE}/initial_object/object_local.ply"
    --export_ply_path "${out_root}/refined_local.ply"
    --output_root "${out_root}"
    --artifixer3d_steps "${DISTILL_STEPS}"
    --config_name "${config_name}"
    --phases distill,render
  )
  if [[ "${FORCE_DISTILL}" == "1" ]]; then
    cmd+=(--replace)
  fi
  if [[ "${DISTILL_MODE}" == "region" && -n "${DISTILL_GAUSSIAN_WEIGHTS:-}" ]]; then
    cmd+=(--local_gaussian_weights_path "${DISTILL_GAUSSIAN_WEIGHTS}")
  fi
  if awk "BEGIN {exit !(${DISTILL_RECON_ETA} > 0)}"; then
    cmd+=(--lambda_reconlosses_override "${DISTILL_RECON_ETA}" --normalize_recon_to_foreground)
  fi

  log "distill ${DISTILL_MODE} strategy=${DISTILL_STRATEGY} eta=${DISTILL_RECON_ETA} steps=${DISTILL_STEPS} -> ${out_root}"
  art_register_container_trap "${container_name}"
  docker run --rm \
    --name "${container_name}" \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="${DISTILL_GPU}" \
    --ipc=host --shm-size=16g \
    --entrypoint bash \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e PYTHONPATH=/workspace/artifixer \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e TORCH_HOME=/cache/torch \
    -e HF_HOME=/cache/huggingface \
    -e XDG_CACHE_HOME=/cache/xdg \
    -e NCCL_P2P_DISABLE=1 \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v "${REPO_ROOT}":/workspace/artifixer \
    -v "${CACHE_HOST}":/cache \
    -w /workspace/artifixer \
    "${DOCKER_IMAGE}" \
    -lc "$(printf '%q ' "${cmd[@]}")"
}

run_distill_eval() {
  local out_root render_dir opacity_dir
  out_root=$(distill_output_root)
  render_dir=$(distill_render_dir)
  opacity_dir=${out_root}/recon_results/${SCENE_FRONT}/artifixer3d/${SCENE_FRONT}/ours_${DISTILL_STEPS}/opacity
  select_v6_map_dirs
  [[ -d "${render_dir}" ]] || { echo "Missing distilled renders: ${render_dir}" >&2; exit 1; }
  [[ -f "${out_root}/refined_local.ply" ]] || { echo "Missing refined_local.ply: ${out_root}/refined_local.ply" >&2; exit 1; }
  log "distill_eval -> ${out_root}/distill_eval.json"
  run_distill_python eval \
    --pred_dir "${DISTILL_PRED_DIR}" \
    --initial_render_dir "${RENDER_DIR}" \
    --distill_render_dir "${render_dir}" \
    --opacity_dir "${OPACITY_DIR}" \
    --distill_opacity_dir "${opacity_dir}" \
    --disorder_dir "${TEXTURE_DISORDER_DIR}" \
    --roi_dir "${FULL_TANK_ROI}" \
    --missing_dir "${REGION_MISSING_DIR}" \
    --frame_indices "$(distill_frame_indices)" \
    --initial_ply "${PREPARED_SCENE}/initial_object/object_local.ply" \
    --refined_ply "${out_root}/refined_local.ply" \
    --output_json "${out_root}/distill_eval.json"
}

run_distill_import() {
  local out_root
  out_root=$(distill_output_root)
  [[ -f "${out_root}/refined_local.ply" ]] || { echo "Missing refined_local.ply: ${out_root}/refined_local.ply" >&2; exit 1; }
  log "distill_import -> ${out_root}/imported"
  run_distill_python import \
    --refined_local_ply "${out_root}/refined_local.ply" \
    --normalization "${PREPARED_SCENE}/normalization.json" \
    --initial_ply "${PREPARED_SCENE}/initial_object/object_local.ply" \
    --output_dir "${out_root}/imported"
}

run_distill_mv() {
  select_v6_map_dirs
  local output_dir=${PREPARED_SCENE}/pred_multiview_support_v6a
  local r10_renders=${PREPARED_SCENE}/artifixer3d_v6a_pred_gs_r10_5k/recon_results/${SCENE_FRONT}/artifixer3d/${SCENE_FRONT}/ours_5000/renders
  local extra=()
  if [[ -d "${r10_renders}" ]]; then
    extra+=(--distill_render_dir "${r10_renders}")
  fi
  log "distill_mv missing∩ROI support -> ${output_dir}"
  PYTHONPATH="${ANALYSIS_PYTHONPATH}" "${WORLDACT_PYTHON}" -m data_processing.pred_multiview_support \
    --transforms_path "${PREPARED_SCENE}/transforms.json" \
    --pred_dir "${DISTILL_PRED_DIR}" \
    --missing_dir "${REGION_MISSING_DIR}" \
    --roi_dir "${FULL_TANK_ROI}" \
    --initial_ply "${PREPARED_SCENE}/initial_object/object_local.ply" \
    --output_dir "${output_dir}" \
    --frame_indices "$(distill_frame_indices)" \
    --stride "${DISTILL_MV_STRIDE:-2}" \
    "${extra[@]}"
}

case "${RUN_STAGE}" in
  prepare)
    prepare_all
    ;;
  prepare_package|prepare_scene|regenerate|install_caption)
    run_front_stage "${RUN_STAGE}"
    ;;
  analyze|analysis)
    analyze_all
    ;;
  infer)
    run_v6a_inference
    ;;
  all)
    prepare_all
    analyze_all
    run_v6a_inference
    ;;
  check)
    print_status
    ;;
  distill_check)
    run_distill_check
    ;;
  distill_smoke)
    run_distill 1000
    ;;
  distill)
    run_distill
    ;;
  distill_eval)
    run_distill_eval
    ;;
  distill_mv)
    run_distill_mv
    ;;
  distill_import)
    run_distill_import
    ;;
  *)
    echo "Unknown stage: ${RUN_STAGE}" >&2
    usage >&2
    exit 2
    ;;
esac
