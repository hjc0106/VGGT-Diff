#!/usr/bin/env bash
# End-to-end ArtiFixer front_views 2D repair pipeline for one parking-lot object.
# Docs: docs/artifixer_front_views_pipeline.md
#
# Usage (repo root):
#   ./scripts/run_artifixer_front_views_pipeline.sh
#   CAPTION_MODE=openai STAGE=caption,install_caption,infer ./scripts/run_artifixer_front_views_pipeline.sh
#   OBJECT_ID=4 SCENE_ID=black_suv_4 PLY=.../object_4_*_final.ply ./scripts/run_artifixer_front_views_pipeline.sh
#
# STAGE: all | prepare_package | caption | prepare_scene | regenerate | install_caption | infer
# CAPTION_MODE: openai (default) | text
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
# shellcheck source=run_art_common.sh
source "${REPO_ROOT}/run_art_common.sh"

# ---- defaults (object_2 / white sedan) ----
OBJECT_ID=${OBJECT_ID:-2}
SCENE_ID=${SCENE_ID:-white_car_2}
SCENE_FRONT=${SCENE_FRONT:-${SCENE_ID}_front_views}
STAGE_DIR=${STAGE_DIR:-worldact/outputs/parking_lot_best/stage1}
PLY=${PLY:-worldact/outputs/parking_lot_best/stage2/object_2_white_car_in_vehicle_cluster_final.ply}
PACKAGE_OUT=${PACKAGE_OUT:-worldact/outputs/parking_lot_best/artifixer_preview_stage2_object_${OBJECT_ID}}
PACKAGE_DIR=${PACKAGE_DIR:-${PACKAGE_OUT}/object_${OBJECT_ID}/object_package}
PREPARED_ROOT=${PREPARED_ROOT:-worldact/outputs/parking_lot_best/artifixer_prepared_stage2}
CAPTION_DIR=${CAPTION_DIR:-${PREPARED_ROOT}/_captions/${SCENE_ID}}
CAPTION_MODE=${CAPTION_MODE:-openai} # openai | text
CAPTION_NAME=${CAPTION_NAME:-}       # default depends on CAPTION_MODE
CONFIG=${CONFIG:-worldact/configs/artifixer/car_front_views.yaml}
PROMPT_TEMPLATE_FILE=${PROMPT_TEMPLATE_FILE:-data_processing/captioning/templates/vehicle_studio.txt}
OPENAI_ENV_FILE=${OPENAI_ENV_FILE:-}
WAN_MODEL_HOST=${WAN_MODEL_HOST:-/models/Wan2.1-T2V-14B-Diffusers}
ARTIFIXER_CHECKPOINT_HOST=${ARTIFIXER_CHECKPOINT_HOST:-/models/artifixer-14b.pt}
CACHE_HOST=${CACHE_HOST:-/tmp/artifixer-cache}
DOCKER_IMAGE=${DOCKER_IMAGE:-artifixer:cuda13-gsplat}

CAPTION_GPU=${CAPTION_GPU:-4}
PREPARE_GPU=${PREPARE_GPU:-${CAPTION_GPU}}
REGEN_GPU=${REGEN_GPU:-${CAPTION_GPU}}
INFER_GPUS=${INFER_GPUS:-4,5,6,7}
INFER_NPROC_PER_NODE=${INFER_NPROC_PER_NODE:-4}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-4}

# Front-view trajectory controls. Defaults preserve the existing isolated-object
# trajectory; scene-local repair can reduce EDGE_MARGIN to move the same ring closer.
FRONT_VIEWS_RING_ELEVATION_DEG=${FRONT_VIEWS_RING_ELEVATION_DEG:-12}
FRONT_VIEWS_CIRCUMRADIUS_SCALE=${FRONT_VIEWS_CIRCUMRADIUS_SCALE:-1.5}
FRONT_VIEWS_LOOK_AT_UP_BIAS=${FRONT_VIEWS_LOOK_AT_UP_BIAS:-0.08}
FRONT_VIEWS_EDGE_MARGIN=${FRONT_VIEWS_EDGE_MARGIN:-0.12}
FRONT_VIEWS_CAMERA_DISTANCE_SCALE=${FRONT_VIEWS_CAMERA_DISTANCE_SCALE:-3.0}
FRONT_VIEWS_CAMERA_DISTANCE_SCALE_MIN=${FRONT_VIEWS_CAMERA_DISTANCE_SCALE_MIN:-2.9}
FRONT_VIEWS_CAMERA_DISTANCE_SCALE_MAX=${FRONT_VIEWS_CAMERA_DISTANCE_SCALE_MAX:-3.2}
# Optional explicit object up-axis (three space-separated values). CARLA
# packages already carry a right-handed +Z-up frame and do not have a
# parking-lot Stage-1 background from which to estimate this axis.
FRONT_VIEWS_UP_AXIS=${FRONT_VIEWS_UP_AXIS:-}
KEEP_SOURCE_ANCHORS=${KEEP_SOURCE_ANCHORS:-0}
ALLOW_ALPHA_EDGE_CONTACT=${ALLOW_ALPHA_EDGE_CONTACT:-0}
EMIT_EFFECTIVE_GAUSSIAN_HITS=${EMIT_EFFECTIVE_GAUSSIAN_HITS:-0}
EMIT_PROJECTED_GAUSSIAN_RESOLUTION=${EMIT_PROJECTED_GAUSSIAN_RESOLUTION:-0}
EMIT_GAUSSIAN_HITS=${EMIT_GAUSSIAN_HITS:-0}
EMIT_VISIBLE_GAUSSIAN_HITS=${EMIT_VISIBLE_GAUSSIAN_HITS:-0}

# Latent/background preservation controls for inference. opacity keeps the
# legacy behavior; object_roi derives an isolated-object ROI from opacity;
# repair_priority uses a continuous per-frame score inside that object ROI;
# region_mask loads explicit per-frame local edit masks;
# texture_disorder_train_aligned restores training opacity noise and adds a
# coverage-gated low-precision residual. Default remains opacity.
LATENT_PRESERVE_MODE=${LATENT_PRESERVE_MODE:-opacity}
LATENT_ROI_ALPHA_THRESHOLD=${LATENT_ROI_ALPHA_THRESHOLD:-0.01}
LATENT_ROI_DILATE_PIXELS=${LATENT_ROI_DILATE_PIXELS:-16}
LATENT_ROI_FEATHER_PIXELS=${LATENT_ROI_FEATHER_PIXELS:-4}
REPAIR_PRIORITY_DIR=${REPAIR_PRIORITY_DIR:-}
REPAIR_PRIORITY_STRENGTH=${REPAIR_PRIORITY_STRENGTH:-1.0}
REPAIR_PRIORITY_PRESERVE_THRESHOLD=${REPAIR_PRIORITY_PRESERVE_THRESHOLD:-0.01}
REPAIR_PRIORITY_LOW_REFERENCE=${REPAIR_PRIORITY_LOW_REFERENCE:-0.35}
REPAIR_PRIORITY_MISSING_REFERENCE=${REPAIR_PRIORITY_MISSING_REFERENCE:-0.85}
REPAIR_PRIORITY_LATENT_LOW_STRENGTH=${REPAIR_PRIORITY_LATENT_LOW_STRENGTH:-0.65}
REPAIR_PRIORITY_LATENT_MISSING_STRENGTH=${REPAIR_PRIORITY_LATENT_MISSING_STRENGTH:-0.90}
REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH=${REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH:-0.80}
REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH=${REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH:-1.0}
TEXTURE_DISORDER_DIR=${TEXTURE_DISORDER_DIR:-}
REGION_MISSING_DIR=${REGION_MISSING_DIR:-}
DISORDER_EXTRA_NOISE_SCALE=${DISORDER_EXTRA_NOISE_SCALE:-0.15}
DISORDER_EXTRA_NOISE_CAP=${DISORDER_EXTRA_NOISE_CAP:-0.15}
DISORDER_PRESERVE_THRESHOLD=${DISORDER_PRESERVE_THRESHOLD:-0.02}
DISORDER_SOLID_OPACITY=${DISORDER_SOLID_OPACITY:-0.5}
DISORDER_SUPPORT_THRESHOLD=${DISORDER_SUPPORT_THRESHOLD:-0.10}
DISORDER_SUPPORT_GATE_START=${DISORDER_SUPPORT_GATE_START:-0.10}
DISORDER_SUPPORT_GATE_FULL=${DISORDER_SUPPORT_GATE_FULL:-0.35}
DISORDER_GAMMA=${DISORDER_GAMMA:-1.5}
DISORDER_GATE_HARD_PRESERVE=${DISORDER_GATE_HARD_PRESERVE:-0}
DISORDER_SPILL_CAP=${DISORDER_SPILL_CAP:-0.03}
DISORDER_SOFT_PRESERVE=${DISORDER_SOFT_PRESERVE:-0}
DISORDER_HIGH_PRECISION_OPACITY=${DISORDER_HIGH_PRECISION_OPACITY:-0.8}
ROI_ZOOM=${ROI_ZOOM:-0}
ROI_ZOOM_CONTEXT_SCALE=${ROI_ZOOM_CONTEXT_SCALE:-1.15}
ROI_ZOOM_MIN_SIDE=${ROI_ZOOM_MIN_SIDE:-64}
DISORDER_HIGH_CORE_ERODE=${DISORDER_HIGH_CORE_ERODE:-4}
DISORDER_TRANSITION_RING=${DISORDER_TRANSITION_RING:-8}
DISORDER_CORE_PRESERVE_SCHEDULE=${DISORDER_CORE_PRESERVE_SCHEDULE:-}
DISORDER_RING_PRESERVE_SCHEDULE=${DISORDER_RING_PRESERVE_SCHEDULE:-}
INFERENCE_SEED=${INFERENCE_SEED:-0}
INFERENCE_STEPS=${INFERENCE_STEPS:-4}
MASTER_PORT=${MASTER_PORT:-29611}
VERIFY_PYTHON=${VERIFY_PYTHON:-${HOME}/miniconda3/envs/worldact/bin/python}
REGION_MASK_DIR=${REGION_MASK_DIR:-}
REGION_MASK_THRESHOLD=${REGION_MASK_THRESHOLD:-0.5}
REGION_MASK_DILATE_PIXELS=${REGION_MASK_DILATE_PIXELS:-0}
REGION_MASK_FEATHER_PIXELS=${REGION_MASK_FEATHER_PIXELS:-4}
REGION_MASK_EDIT_STRENGTH=${REGION_MASK_EDIT_STRENGTH:-1.0}
HARD_PRESERVE_BACKGROUND=${HARD_PRESERVE_BACKGROUND:-1}
COMPOSITE_OUTSIDE_ROI=${COMPOSITE_OUTSIDE_ROI:-0}
SAVE_LATENT_MASK_DIAGNOSTICS=${SAVE_LATENT_MASK_DIAGNOSTICS:-0}

# Hand-written fallback (used when CAPTION_MODE=text). Includes strict black-void constraints.
CAPTION_TEXT=${CAPTION_TEXT:-"A clean studio multi-view rendering of a single white sedan with four doors, four wheels, windows, headlights, and a complete closed body. Isolated single car on a pure black background, no floor, no ground plane, no contact shadow, no reflection, no environment, no other objects. The complete vehicle is centered and floating in void black space, with consistent white paint, windows, tires, body shape, and details across all surrounding views. No other vehicles, people, road, parking lot, buildings, text, scenery, ground, shadow, or reflection."}

STAGE=${STAGE:-all}

if [[ -z "${CAPTION_NAME}" ]]; then
  if [[ "${CAPTION_MODE}" == "openai" ]]; then
    CAPTION_NAME=caption_openai_strict.h5
  else
    CAPTION_NAME=caption.h5
  fi
fi
CAPTION_H5=${CAPTION_H5:-${CAPTION_DIR}/${CAPTION_NAME}}
CAPTION_STEM=${CAPTION_NAME%.h5}
SAVE_DIR=${SAVE_DIR:-worldact/outputs/parking_lot_best/artifixer_2d_stage2_front_views_v2_${CAPTION_STEM}}

# Prefer openai.env, then packy.env, unless OPENAI_ENV_FILE is set.
if [[ -z "${OPENAI_ENV_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/openai.env" ]]; then
    OPENAI_ENV_FILE="${REPO_ROOT}/openai.env"
  elif [[ -f "${REPO_ROOT}/packy.env" ]]; then
    OPENAI_ENV_FILE="${REPO_ROOT}/packy.env"
  fi
fi

log() { echo "[pipeline] $*"; }

want_stage() {
  local name=$1
  [[ "$STAGE" == "all" ]] && return 0
  [[ ",${STAGE}," == *",${name},"* ]] && return 0
  return 1
}

docker_base() {
  # usage: docker_base <host_gpu_list> <bash -lc script> [extra docker args...]
  local gpus=$1
  local script=$2
  shift 2
  docker run --rm \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="${gpus}" \
    --ipc=host \
    --shm-size=16g \
    --entrypoint bash \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e USER="$(id -un)" \
    -e LOGNAME="$(id -un)" \
    -e PYTHONPATH=/workspace/artifixer \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e TORCH_HOME=/cache/torch \
    -e HF_HOME=/cache/huggingface \
    -e XDG_CACHE_HOME=/cache/xdg \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_SHM_DISABLE=0 \
    -e NCCL_CROSS_NIC=1 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$REPO_ROOT":/workspace/artifixer \
    -v "$CACHE_HOST":/cache \
    -v "$WAN_MODEL_HOST":/models/Wan2.1-T2V-14B-Diffusers:ro \
    "$@" \
    -w /workspace/artifixer \
    "$DOCKER_IMAGE" \
    -lc "$script"
}

verify_caption_h5() {
  local path=$1
  "${VERIFY_PYTHON}" - <<PY
import h5py, numpy as np
from pathlib import Path
p = Path("${path}")
with h5py.File(p, "r") as f:
    d = f[list(f.keys())[0]]
    u = np.asarray(d[:], np.uint16)
    cap = d.attrs.get("caption", "")
    if isinstance(cap, bytes):
        cap = cap.decode()
bf = (u.astype(np.uint32) << 16).view(np.float32)
std, amin, amax = float(bf.std()), float(bf.min()), float(bf.max())
print(f"caption: {cap[:160]}{'...' if len(cap) > 160 else ''}")
print(f"bf16_check tokens={bf.shape[0]} std={std:.4f} min={amin:.4f} max={amax:.4f}")
if std >= 1.0 or abs(amin) >= 10 or abs(amax) >= 10:
    raise SystemExit("caption embedding scale looks corrupt")
print("caption OK:", p)
PY
}

stage_prepare_package() {
  log "① prepare package  object_id=${OBJECT_ID}  ply=${PLY}"
  [[ -f "$PLY" ]] || { echo "missing PLY: $PLY" >&2; exit 1; }
  [[ -d "$STAGE_DIR" ]] || { echo "missing stage-dir: $STAGE_DIR" >&2; exit 1; }

  if [[ -x "${HOME}/miniconda3/envs/worldact/bin/python" ]]; then
    PYTHONPATH="${REPO_ROOT}/worldact:${PYTHONPATH:-}" \
      "${HOME}/miniconda3/envs/worldact/bin/python" \
      worldact/scripts/run_artifixer_refinement.py \
      --ply "$PLY" \
      --stage-dir "$STAGE_DIR" \
      --object-id "$OBJECT_ID" \
      --output-dir "$PACKAGE_OUT" \
      --phase prepare \
      --config "$CONFIG"
  else
    docker_base "$PREPARE_GPU" "
      python worldact/scripts/run_artifixer_refinement.py \
        --ply ${PLY} \
        --stage-dir ${STAGE_DIR} \
        --object-id ${OBJECT_ID} \
        --output-dir ${PACKAGE_OUT} \
        --phase prepare \
        --config ${CONFIG}
    "
  fi
  [[ -d "$PACKAGE_DIR" ]] || { echo "package missing: $PACKAGE_DIR" >&2; exit 1; }
  log "package ok: $PACKAGE_DIR"
}

stage_caption_text() {
  log "② encode caption (text, SINGLE GPU=${CAPTION_GPU}) -> ${CAPTION_H5}"
  mkdir -p "${CAPTION_DIR}"
  local prompt_file="${CAPTION_DIR}/prompt.txt"
  printf '%s' "$CAPTION_TEXT" >"$prompt_file"

  docker_base "$CAPTION_GPU" "
      python - <<'PY'
from pathlib import Path
from data_processing.captioning.generate_captions import write_caption_embedding_hdf5
prompt = Path('${prompt_file}').read_text().strip()
info = write_caption_embedding_hdf5(
    Path('${CAPTION_H5}'),
    prompt,
    text_encoder_model_id='/models/Wan2.1-T2V-14B-Diffusers',
)
print(info)
PY
  "
  verify_caption_h5 "$CAPTION_H5"
}

stage_caption_openai() {
  log "② OpenAI template+anchors -> Wan encode (SINGLE GPU=${CAPTION_GPU}) -> ${CAPTION_H5}"
  [[ -d "$PACKAGE_DIR/anchors" ]] || { echo "missing anchors: ${PACKAGE_DIR}/anchors (run prepare_package)" >&2; exit 1; }
  [[ -f "$PROMPT_TEMPLATE_FILE" ]] || { echo "missing template: $PROMPT_TEMPLATE_FILE" >&2; exit 1; }
  if [[ -z "${OPENAI_ENV_FILE}" || ! -f "${OPENAI_ENV_FILE}" ]]; then
    echo "missing OpenAI env file (set OPENAI_ENV_FILE=./openai.env or ./packy.env)" >&2
    exit 1
  fi

  mkdir -p "${CAPTION_DIR}"
  local prompt_out="${CAPTION_DIR}/${CAPTION_STEM%.h5}.txt"
  # shellcheck disable=SC1090
  set -a
  # shellcheck source=/dev/null
  source "${OPENAI_ENV_FILE}"
  set +a

  docker_base "$CAPTION_GPU" "
      python -m data_processing.captioning.generate_captions \
        --input_path ${PACKAGE_DIR}/anchors \
        --output_path ${CAPTION_H5} \
        --caption_backend openai \
        --prompt_template_file ${PROMPT_TEMPLATE_FILE} \
        --save_prompt_path ${prompt_out} \
        --text_encoder_model_id /models/Wan2.1-T2V-14B-Diffusers
    " \
    -e "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
    -e "OPENAI_API_BASE=${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}" \
    -e "OPENAI_MODEL=${OPENAI_MODEL:-}" \
    -e "OPENAI_MAX_OUTPUT_TOKENS=${OPENAI_MAX_OUTPUT_TOKENS:-2048}" \
    -e "OPENAI_REASONING_EFFORT=${OPENAI_REASONING_EFFORT:-low}"

  verify_caption_h5 "$CAPTION_H5"
  log "wrote prompt text: $prompt_out"
}

stage_caption() {
  case "${CAPTION_MODE}" in
    openai) stage_caption_openai ;;
    text) stage_caption_text ;;
    *) echo "unknown CAPTION_MODE=${CAPTION_MODE} (use openai|text)" >&2; exit 1 ;;
  esac
}

stage_prepare_scene() {
  log "③ prepare scene  scene_id=${SCENE_ID}"
  [[ -d "$PACKAGE_DIR" ]] || { echo "missing package: $PACKAGE_DIR" >&2; exit 1; }
  [[ -f "$CAPTION_H5" ]] || { echo "missing caption: $CAPTION_H5 (run STAGE=caption)" >&2; exit 1; }

  local gaussian_hits_arg
  if [[ "${EMIT_GAUSSIAN_HITS}" == "1" ]]; then
    gaussian_hits_arg="--emit_gaussian_hits"
  else
    gaussian_hits_arg=""
  fi
  local visible_gaussian_hits_arg
  if [[ "${EMIT_VISIBLE_GAUSSIAN_HITS}" == "1" ]]; then
    visible_gaussian_hits_arg="--emit_visible_gaussian_hits"
  else
    visible_gaussian_hits_arg=""
  fi
  local effective_gaussian_hits_arg
  if [[ "${EMIT_EFFECTIVE_GAUSSIAN_HITS}" == "1" ]]; then
    effective_gaussian_hits_arg="--emit_effective_gaussian_hits"
  else
    effective_gaussian_hits_arg=""
  fi
  local projected_resolution_arg
  if [[ "${EMIT_PROJECTED_GAUSSIAN_RESOLUTION}" == "1" ]]; then
    projected_resolution_arg="--emit_projected_gaussian_resolution"
  else
    projected_resolution_arg=""
  fi
  docker_base "$PREPARE_GPU" "
    python -m data_processing.worldact_object_inputs \
      --package_dir ${PACKAGE_DIR} \
      --output_root ${PREPARED_ROOT} \
      --scene_id ${SCENE_ID} \
      --caption_h5 ${CAPTION_H5} \
      --target_render_device cuda \
      ${gaussian_hits_arg} \
      ${visible_gaussian_hits_arg} \
      ${effective_gaussian_hits_arg} \
      ${projected_resolution_arg}
  "
  [[ -f "${PREPARED_ROOT}/${SCENE_ID}/split.json" ]] || {
    echo "prepare scene failed: no split.json" >&2
    exit 1
  }
  log "prepared scene ok: ${PREPARED_ROOT}/${SCENE_ID}"
}

stage_regenerate() {
  local up_axis_arg=""
  if [[ -n "${FRONT_VIEWS_UP_AXIS}" ]]; then
    read -r -a up_axis_values <<<"${FRONT_VIEWS_UP_AXIS}"
    if [[ "${#up_axis_values[@]}" -ne 3 ]]; then
      echo "FRONT_VIEWS_UP_AXIS must contain exactly three values" >&2
      exit 1
    fi
    up_axis_arg="--up-axis ${up_axis_values[0]} ${up_axis_values[1]} ${up_axis_values[2]}"
  fi
  local keep_source_anchors_arg=""
  if [[ "${KEEP_SOURCE_ANCHORS}" == "1" ]]; then
    keep_source_anchors_arg="--keep-source-anchors"
  fi
  local trajectory_qc_arg
  if [[ "${ALLOW_ALPHA_EDGE_CONTACT}" == "1" ]]; then
    trajectory_qc_arg="--allow-alpha-edge-contact"
  else
    trajectory_qc_arg="--no-allow-alpha-edge-contact"
  fi
  local gaussian_hits_arg
  if [[ "${EMIT_GAUSSIAN_HITS}" == "1" ]]; then
    gaussian_hits_arg="--emit_gaussian_hits"
  else
    gaussian_hits_arg=""
  fi
  local visible_gaussian_hits_arg
  if [[ "${EMIT_VISIBLE_GAUSSIAN_HITS}" == "1" ]]; then
    visible_gaussian_hits_arg="--emit_visible_gaussian_hits"
  else
    visible_gaussian_hits_arg=""
  fi
  local effective_gaussian_hits_arg
  if [[ "${EMIT_EFFECTIVE_GAUSSIAN_HITS}" == "1" ]]; then
    effective_gaussian_hits_arg="--emit_effective_gaussian_hits"
  else
    effective_gaussian_hits_arg=""
  fi
  local projected_resolution_arg
  if [[ "${EMIT_PROJECTED_GAUSSIAN_RESOLUTION}" == "1" ]]; then
    projected_resolution_arg="--emit_projected_gaussian_resolution"
  else
    projected_resolution_arg=""
  fi
  log "④ regenerate front_views -> ${SCENE_FRONT}"
  log "trajectory: elevation=${FRONT_VIEWS_RING_ELEVATION_DEG} edge_margin=${FRONT_VIEWS_EDGE_MARGIN} circumradius_scale=${FRONT_VIEWS_CIRCUMRADIUS_SCALE} allow_edge_contact=${ALLOW_ALPHA_EDGE_CONTACT}"
  docker_base "$REGEN_GPU" "
    python -u worldact/scripts/regenerate_object_trajectory.py \
      --source-scene ${PREPARED_ROOT}/${SCENE_ID} \
      --output-scene ${PREPARED_ROOT}/${SCENE_FRONT} \
      --mode front_views \
      ${up_axis_arg} \
      --stage-dir ${STAGE_DIR} \
      --ring-elevation-deg ${FRONT_VIEWS_RING_ELEVATION_DEG} \
      --circumradius-scale ${FRONT_VIEWS_CIRCUMRADIUS_SCALE} \
      --look-at-up-bias ${FRONT_VIEWS_LOOK_AT_UP_BIAS} \
      --edge-margin ${FRONT_VIEWS_EDGE_MARGIN} \
      --camera-distance-scale ${FRONT_VIEWS_CAMERA_DISTANCE_SCALE} \
      --camera-distance-scale-range ${FRONT_VIEWS_CAMERA_DISTANCE_SCALE_MIN} ${FRONT_VIEWS_CAMERA_DISTANCE_SCALE_MAX} \
      ${trajectory_qc_arg} \
      ${keep_source_anchors_arg} \
      ${gaussian_hits_arg} \
      ${visible_gaussian_hits_arg} \
      ${effective_gaussian_hits_arg} \
      ${projected_resolution_arg} \
      --replace
  "
  local qc="${PREPARED_ROOT}/${SCENE_FRONT}/trajectory_qc.json"
  [[ -f "$qc" ]] || { echo "missing QC: $qc" >&2; exit 1; }
  python3 - <<PY
import json
from pathlib import Path
qc = json.loads(Path("${qc}").read_text())
assert qc.get("passed"), qc
print("QC passed; preview:", Path("${PREPARED_ROOT}/${SCENE_FRONT}/trajectory_preview.jpg"))
PY
  log "regenerate ok: ${PREPARED_ROOT}/${SCENE_FRONT}"
}

stage_install_caption() {
  # Copy CAPTION_H5 into front_views scene and point split.json prompt_path at it.
  log "④b install caption into ${SCENE_FRONT}"
  [[ -f "$CAPTION_H5" ]] || { echo "missing caption: $CAPTION_H5" >&2; exit 1; }
  local front_root="${PREPARED_ROOT}/${SCENE_FRONT}"
  [[ -d "$front_root" ]] || { echo "missing front_views scene: $front_root (run regenerate)" >&2; exit 1; }

  python3 - <<PY
import json, shutil
from pathlib import Path
src = Path("${CAPTION_H5}")
front = Path("${front_root}")
scene_id = "${SCENE_FRONT}"
name = "${CAPTION_NAME}"
dst = front / "captions" / scene_id / name
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
split_path = front / "split.json"
split = json.loads(split_path.read_text())
entry = split["test"][scene_id]
rel = f"captions/{scene_id}/{name}"
entry["prompt_path"] = rel
split_path.write_text(json.dumps(split, indent=2) + "\n")
print("installed", dst)
print("prompt_path ->", rel)
PY
}

stage_infer() {
  local latent_mask_args="--latent_preserve_mode ${LATENT_PRESERVE_MODE} --latent_roi_alpha_threshold ${LATENT_ROI_ALPHA_THRESHOLD} --latent_roi_dilate_pixels ${LATENT_ROI_DILATE_PIXELS} --latent_roi_feather_pixels ${LATENT_ROI_FEATHER_PIXELS}"
  local repair_priority_dir=""
  if [[ "${LATENT_PRESERVE_MODE}" == "repair_priority" ]]; then
    repair_priority_dir="${REPAIR_PRIORITY_DIR:-${PREPARED_ROOT}/${SCENE_FRONT}/recon_results/${SCENE_FRONT}/reconstruction/${SCENE_FRONT}/ours_0/repair_priority}"
    [[ -d "${repair_priority_dir}" ]] || { echo "missing repair priority dir: ${repair_priority_dir}" >&2; exit 1; }
    latent_mask_args+=" --repair_priority_dir ${repair_priority_dir} --repair_priority_strength ${REPAIR_PRIORITY_STRENGTH} --repair_priority_preserve_threshold ${REPAIR_PRIORITY_PRESERVE_THRESHOLD} --repair_priority_low_reference ${REPAIR_PRIORITY_LOW_REFERENCE} --repair_priority_missing_reference ${REPAIR_PRIORITY_MISSING_REFERENCE} --repair_priority_latent_low_strength ${REPAIR_PRIORITY_LATENT_LOW_STRENGTH} --repair_priority_latent_missing_strength ${REPAIR_PRIORITY_LATENT_MISSING_STRENGTH} --repair_priority_composite_low_strength ${REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH} --repair_priority_composite_missing_strength ${REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH}"
  fi

  local texture_disorder_dir=""
  if [[ "${LATENT_PRESERVE_MODE}" == "texture_disorder_train_aligned" ]]; then
    texture_disorder_dir="${TEXTURE_DISORDER_DIR}"
    [[ -n "${texture_disorder_dir}" ]] || { echo "TEXTURE_DISORDER_DIR must point at region_low_precision/*.npy for texture_disorder_train_aligned" >&2; exit 1; }
    [[ -d "${texture_disorder_dir}" ]] || { echo "missing texture disorder dir: ${texture_disorder_dir}" >&2; exit 1; }
    [[ -n "${REGION_MISSING_DIR}" ]] || { echo "REGION_MISSING_DIR must point at region_missing/*.npy for texture_disorder_train_aligned" >&2; exit 1; }
    [[ -d "${REGION_MISSING_DIR}" ]] || { echo "missing region-missing dir: ${REGION_MISSING_DIR}" >&2; exit 1; }
    repair_priority_dir="${REPAIR_PRIORITY_DIR}"
    [[ -n "${repair_priority_dir}" ]] || { echo "REPAIR_PRIORITY_DIR must point at diffusion_edit_priority/*.npy for texture_disorder_train_aligned composite weights" >&2; exit 1; }
    [[ -d "${repair_priority_dir}" ]] || { echo "missing repair priority dir: ${repair_priority_dir}" >&2; exit 1; }
    latent_mask_args+=" --texture_disorder_dir ${texture_disorder_dir} --region_missing_dir ${REGION_MISSING_DIR} --disorder_extra_noise_scale ${DISORDER_EXTRA_NOISE_SCALE} --disorder_extra_noise_cap ${DISORDER_EXTRA_NOISE_CAP} --disorder_preserve_threshold ${DISORDER_PRESERVE_THRESHOLD} --disorder_solid_opacity ${DISORDER_SOLID_OPACITY} --disorder_support_threshold ${DISORDER_SUPPORT_THRESHOLD} --disorder_support_gate_start ${DISORDER_SUPPORT_GATE_START} --disorder_support_gate_full ${DISORDER_SUPPORT_GATE_FULL} --disorder_gamma ${DISORDER_GAMMA} --disorder_spill_cap ${DISORDER_SPILL_CAP} --disorder_soft_preserve ${DISORDER_SOFT_PRESERVE} --disorder_high_precision_opacity ${DISORDER_HIGH_PRECISION_OPACITY} --disorder_high_core_erode ${DISORDER_HIGH_CORE_ERODE} --disorder_transition_ring ${DISORDER_TRANSITION_RING} --repair_priority_dir ${repair_priority_dir} --repair_priority_strength ${REPAIR_PRIORITY_STRENGTH} --repair_priority_preserve_threshold ${REPAIR_PRIORITY_PRESERVE_THRESHOLD} --repair_priority_low_reference ${REPAIR_PRIORITY_LOW_REFERENCE} --repair_priority_missing_reference ${REPAIR_PRIORITY_MISSING_REFERENCE} --repair_priority_latent_low_strength ${REPAIR_PRIORITY_LATENT_LOW_STRENGTH} --repair_priority_latent_missing_strength ${REPAIR_PRIORITY_LATENT_MISSING_STRENGTH} --repair_priority_composite_low_strength ${REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH} --repair_priority_composite_missing_strength ${REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH}"
    if [[ "${ROI_ZOOM}" == "1" ]]; then
      latent_mask_args+=" --roi_zoom --roi_zoom_context_scale ${ROI_ZOOM_CONTEXT_SCALE} --roi_zoom_min_side ${ROI_ZOOM_MIN_SIDE}"
    else
      latent_mask_args+=" --no-roi_zoom"
    fi
    if [[ -n "${DISORDER_CORE_PRESERVE_SCHEDULE}" ]]; then
      latent_mask_args+=" --disorder_core_preserve_schedule ${DISORDER_CORE_PRESERVE_SCHEDULE}"
    fi
    if [[ -n "${DISORDER_RING_PRESERVE_SCHEDULE}" ]]; then
      latent_mask_args+=" --disorder_ring_preserve_schedule ${DISORDER_RING_PRESERVE_SCHEDULE}"
    fi
    if [[ "${DISORDER_GATE_HARD_PRESERVE}" == "1" ]]; then
      latent_mask_args+=" --disorder_gate_hard_preserve"
    else
      latent_mask_args+=" --no-disorder_gate_hard_preserve"
    fi
  fi

  local region_mask_dir=""
  if [[ "${LATENT_PRESERVE_MODE}" == "region_mask" ]]; then
    region_mask_dir="${REGION_MASK_DIR:-${PREPARED_ROOT}/${SCENE_FRONT}/edit_masks}"
    [[ -d "${region_mask_dir}" ]] || { echo "missing region mask dir: ${region_mask_dir}" >&2; exit 1; }
    latent_mask_args+=" --region_mask_dir ${region_mask_dir} --region_mask_threshold ${REGION_MASK_THRESHOLD} --region_mask_dilate_pixels ${REGION_MASK_DILATE_PIXELS} --region_mask_feather_pixels ${REGION_MASK_FEATHER_PIXELS} --region_mask_edit_strength ${REGION_MASK_EDIT_STRENGTH}"
  fi

  if [[ "${HARD_PRESERVE_BACKGROUND}" == "1" ]]; then
    latent_mask_args+=" --hard_preserve_background"
  else
    latent_mask_args+=" --no-hard_preserve_background"
  fi
  if [[ "${COMPOSITE_OUTSIDE_ROI}" == "1" ]]; then
    latent_mask_args+=" --composite_outside_roi"
  fi
  if [[ "${SAVE_LATENT_MASK_DIAGNOSTICS}" == "1" ]]; then
    latent_mask_args+=" --save_latent_mask_diagnostics"
  fi

  log "⑤ inference on GPUs ${INFER_GPUS}  scene=${SCENE_FRONT}  save=${SAVE_DIR}  steps=${INFERENCE_STEPS}"
  log "latent preserve: mode=${LATENT_PRESERVE_MODE} alpha_threshold=${LATENT_ROI_ALPHA_THRESHOLD} dilate=${LATENT_ROI_DILATE_PIXELS} feather=${LATENT_ROI_FEATHER_PIXELS} hard=${HARD_PRESERVE_BACKGROUND} composite=${COMPOSITE_OUTSIDE_ROI}"
  [[ "${LATENT_PRESERVE_MODE}" != "repair_priority" ]] || log "repair priority: dir=${repair_priority_dir} strength=${REPAIR_PRIORITY_STRENGTH} preserve<=${REPAIR_PRIORITY_PRESERVE_THRESHOLD} latent=${REPAIR_PRIORITY_LATENT_LOW_STRENGTH}/${REPAIR_PRIORITY_LATENT_MISSING_STRENGTH} composite=${REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH}/${REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH}"
  [[ "${LATENT_PRESERVE_MODE}" != "texture_disorder_train_aligned" ]] || log "train-aligned disorder: dir=${texture_disorder_dir} missing_dir=${REGION_MISSING_DIR} extra_scale=${DISORDER_EXTRA_NOISE_SCALE} extra_cap=${DISORDER_EXTRA_NOISE_CAP} support_tau=${DISORDER_SUPPORT_THRESHOLD} gate=${DISORDER_SUPPORT_GATE_START}/${DISORDER_SUPPORT_GATE_FULL} gamma=${DISORDER_GAMMA} spill_cap=${DISORDER_SPILL_CAP} soft_preserve=${DISORDER_SOFT_PRESERVE} high_opacity>=${DISORDER_HIGH_PRECISION_OPACITY} gate_hard=${DISORDER_GATE_HARD_PRESERVE} preserve<=${DISORDER_PRESERVE_THRESHOLD} solid>=${DISORDER_SOLID_OPACITY} composite=${REPAIR_PRIORITY_COMPOSITE_LOW_STRENGTH}/${REPAIR_PRIORITY_COMPOSITE_MISSING_STRENGTH} repair_priority_dir=${repair_priority_dir}"
  [[ "${LATENT_PRESERVE_MODE}" != "region_mask" ]] || log "region mask: dir=${region_mask_dir} threshold=${REGION_MASK_THRESHOLD} dilate=${REGION_MASK_DILATE_PIXELS} feather=${REGION_MASK_FEATHER_PIXELS} strength=${REGION_MASK_EDIT_STRENGTH}"
  art_log_gpu "before-infer"
  local split_path="${PREPARED_ROOT}/${SCENE_FRONT}/split.json"
  [[ -f "$split_path" ]] || { echo "missing split: $split_path" >&2; exit 1; }

  CONTAINER_NAME=${CONTAINER_NAME:-artifixer-run-front-views-$$}
  art_register_container_trap "${CONTAINER_NAME}"
  docker run --rm \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="${INFER_GPUS}" \
    --ipc=host \
    --shm-size=64g \
    --entrypoint bash \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e USER="$(id -un)" \
    -e LOGNAME="$(id -un)" \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e OMP_NUM_THREADS=8 \
    -e MKL_NUM_THREADS=8 \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_SHM_DISABLE=0 \
    -e NCCL_CROSS_NIC=1 \
    -e TORCH_HOME=/cache/torch \
    -e HF_HOME=/cache/huggingface \
    -e XDG_CACHE_HOME=/cache/xdg \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$REPO_ROOT":/workspace/artifixer \
    -v "$ARTIFIXER_CHECKPOINT_HOST":/data/artifixer-checkpoints/artifixer-14b.pt:ro \
    -v "$WAN_MODEL_HOST":/models/Wan2.1-T2V-14B-Diffusers:ro \
    -v "$CACHE_HOST":/cache \
    -w /workspace/artifixer \
    "$DOCKER_IMAGE" \
    -lc "
      $(art_inner_cleanup_preamble)
      python -m torch.distributed.run \\
        --nproc_per_node=${INFER_NPROC_PER_NODE} \\
        --master_port=${MASTER_PORT} \\
        -m model_eval.run_inference \\
        --evalset reconstructed_colmap \\
        --checkpoint_pt /data/artifixer-checkpoints/artifixer-14b.pt \\
        --model_id /models/Wan2.1-T2V-14B-Diffusers \\
        --save_dir ${SAVE_DIR} \\
        --split_path ${split_path} \\
        --scene_id ${SCENE_FRONT} \\
        --render_trajectory trajectory \\
        --inference_pipeline bidirectional \\
        --context_parallel_size ${CONTEXT_PARALLEL_SIZE} \\
        --num_inference_steps ${INFERENCE_STEPS} \\
        --inference_seed ${INFERENCE_SEED} \\
        --distributed_timeout_minutes 60 \\
        --max_neighbors_per_encode 1 \\
        ${latent_mask_args} \\
        --save_frame_outputs_only \\
        --replace_if_exists
    "
  log "infer done -> ${SAVE_DIR}/artifixer-14b/.../${SCENE_FRONT}/frames/batch_0000/pred/"
}

main() {
  log "OBJECT_ID=${OBJECT_ID} SCENE_ID=${SCENE_ID} SCENE_FRONT=${SCENE_FRONT}"
  log "CAPTION_MODE=${CAPTION_MODE} CAPTION_H5=${CAPTION_H5}"
  log "SAVE_DIR=${SAVE_DIR} INFER_GPUS=${INFER_GPUS} CAPTION_GPU=${CAPTION_GPU} INFERENCE_STEPS=${INFERENCE_STEPS}"
  log "LATENT_PRESERVE_MODE=${LATENT_PRESERVE_MODE} HARD_PRESERVE_BACKGROUND=${HARD_PRESERVE_BACKGROUND} COMPOSITE_OUTSIDE_ROI=${COMPOSITE_OUTSIDE_ROI}"
  log "STAGE=${STAGE}"

  want_stage prepare_package && stage_prepare_package
  want_stage caption && stage_caption
  want_stage prepare_scene && stage_prepare_scene
  want_stage regenerate && stage_regenerate
  # After regenerate, install preferred caption into front_views (openai/text file name).
  if want_stage all || want_stage install_caption; then
    if [[ -f "${CAPTION_H5}" && -d "${PREPARED_ROOT}/${SCENE_FRONT}" ]]; then
      stage_install_caption
    elif want_stage install_caption; then
      stage_install_caption
    fi
  fi
  want_stage infer && stage_infer
  log "finished"
}

main "$@"
