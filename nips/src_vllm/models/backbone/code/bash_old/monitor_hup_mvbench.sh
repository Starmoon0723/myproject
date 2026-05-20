#!/usr/bin/env bash
# Run Qwen3-VL vLLM offline evaluation on MVBench.
# Usage:
#   conda activate <env>
#   bash monitor_hup_mvbench.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1 N_PROC=2 bash monitor_hup_mvbench.sh
#   VIDEO_PRUNING_RATE=0.5 VIDEO_PRUNING_METHOD=evs bash monitor_hup_mvbench.sh
#   USE_MM_CACHE=1 MM_CACHE_READ_ONLY=1 bash monitor_hup_mvbench.sh

set -Eeuo pipefail

WORKSPACE_ROOT="/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm"
PROJECT_ROOT="${WORKSPACE_ROOT}/myproject/nips"
SRC_ROOT="${PROJECT_ROOT}/src_vllm"
# use vllm_feature_my_change 源码
VLLM_SRC="${WORKSPACE_ROOT}/vllm_feature_my_change"
ENV_FILE="${WORKSPACE_ROOT}/cache_env_new.sh"

# The env file requires an activated conda env because it configures CUDA_HOME
# and LD_LIBRARY_PATH from CONDA_PREFIX.
source "${ENV_FILE}"

export PYTHONPATH="${VLLM_SRC}:${SRC_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_FLASH_ATTN_VERSION="${VLLM_FLASH_ATTN_VERSION:-2}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/hfmodel/qwen3vl_8b}"
DATASET_NAME="${DATASET_NAME:-MVBench}" # Video-MME，MVBench
N_PROC="${N_PROC:-2}"
GPU_MEMORY="${GPU_MEMORY:-0.9}"
FPS="${FPS:-2}"
NFRAME="${NFRAME:--1}"

CUDA_VISIBLE_DEVICES=0,1

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/results/backbone}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/output/logs/backbone/mvbench}"
INFERENCE_SCRIPT="${SRC_ROOT}/models/run.py"

SAMPLE_PATH="${SAMPLE_PATH:-}"
MM_CACHE_DIR="${MM_CACHE_DIR:-${PROJECT_ROOT}/output/mm_cache}"
USE_MM_CACHE="${USE_MM_CACHE:-0}"
MM_CACHE_READ_ONLY="${MM_CACHE_READ_ONLY:-0}"

VIDEO_PRUNING_RATE="${VIDEO_PRUNING_RATE:-}"
VIDEO_PRUNING_METHOD="${VIDEO_PRUNING_METHOD:-evs}"
VIDEO_DIVPRUNE_EXACT_THRESHOLD="${VIDEO_DIVPRUNE_EXACT_THRESHOLD:-4096}"

DISABLE_RESUME="${DISABLE_RESUME:-0}"
POST_PROCESS="${POST_PROCESS:-0}"
MAX_RETRIES="${MAX_RETRIES:--1}"  # -1 means retry forever.
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: ${LOG_FILE}"
echo "========================================================="
echo "Starting Job at $(date)"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "Dataset: ${DATASET_NAME}"
echo "N_PROC: ${N_PROC}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "GPU memory utilization: ${GPU_MEMORY}"
echo "FPS: ${FPS}, NFRAME: ${NFRAME}"
echo "Video pruning: rate=${VIDEO_PRUNING_RATE:-disabled}, method=${VIDEO_PRUNING_METHOD}"
echo "MM cache: use=${USE_MM_CACHE}, read_only=${MM_CACHE_READ_ONLY}, dir=${MM_CACHE_DIR}"
echo "========================================================="

if [ ! -f "${INFERENCE_SCRIPT}" ]; then
    echo "[ERROR] Inference script not found: ${INFERENCE_SCRIPT}"
    exit 1
fi

if [[ "${MODEL_PATH}" == /* ]] && [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Local model path does not exist: ${MODEL_PATH}"
    exit 1
fi

build_cmd() {
    CMD=(
        python "${INFERENCE_SCRIPT}"
        --nproc "${N_PROC}"
        --model_name "${MODEL_NAME}"
        --model_path "${MODEL_PATH}"
        --dataset "${DATASET_NAME}"
        --output "${OUTPUT_DIR}"
        --gpu_memory_utilization "${GPU_MEMORY}"
        --fps "${FPS}"
        --nframe "${NFRAME}"
        --video_pruning_method "${VIDEO_PRUNING_METHOD}"
        --video_divprune_exact_threshold "${VIDEO_DIVPRUNE_EXACT_THRESHOLD}"
    )

    if [ -n "${SAMPLE_PATH}" ]; then
        CMD+=(--sample_path "${SAMPLE_PATH}")
    fi
    if [ "${USE_MM_CACHE}" = "1" ]; then
        mkdir -p "${MM_CACHE_DIR}"
        CMD+=(--use_mm_cache --mm_cache_dir "${MM_CACHE_DIR}")
    fi
    if [ "${MM_CACHE_READ_ONLY}" = "1" ]; then
        CMD+=(--mm_cache_read_only)
    fi
    if [ -n "${VIDEO_PRUNING_RATE}" ]; then
        CMD+=(--video_pruning_rate "${VIDEO_PRUNING_RATE}")
    fi
    if [ "${POST_PROCESS}" = "1" ]; then
        CMD+=(--post_process)
    fi
    if [ "${DISABLE_RESUME}" = "1" ]; then
        CMD+=(--disable_resume)
    fi
}

RETRY_COUNT=0

while true; do
    build_cmd
    {
        echo "---------------------------------------------------------"
        echo "Starting Inference (Attempt #$((RETRY_COUNT + 1))) at $(date)"
        printf 'Command:'
        printf ' %q' "${CMD[@]}"
        echo
        echo "---------------------------------------------------------"
    } | tee -a "${LOG_FILE}"

    set +e
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ "${EXIT_CODE}" -eq 0 ]; then
        echo "Inference finished successfully at $(date)." | tee -a "${LOG_FILE}"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Process crashed with exit code ${EXIT_CODE} at $(date)." | tee -a "${LOG_FILE}"
    if [ "${MAX_RETRIES}" -ge 0 ] && [ "${RETRY_COUNT}" -gt "${MAX_RETRIES}" ]; then
        echo "Reached MAX_RETRIES=${MAX_RETRIES}; stop retrying." | tee -a "${LOG_FILE}"
        exit "${EXIT_CODE}"
    fi
    echo "Waiting ${RETRY_SLEEP_SECONDS}s before retry..." | tee -a "${LOG_FILE}"
    sleep "${RETRY_SLEEP_SECONDS}"
done

echo "Full log saved to: ${LOG_FILE}"