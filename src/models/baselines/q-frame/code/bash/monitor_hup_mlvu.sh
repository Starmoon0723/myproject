#!/bin/bash
# chmod +x monitor_hup_videomme.sh
set -o pipefail

# -----------------------------
# 日志
# -----------------------------
LOG_DIR="/data/oceanus_share/shangshouduo-jk/myproject/output/logs/baselines/q-frame/longclip/mlvu"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"

echo "📝 Log file will be: $LOG_FILE"
echo "========================================================="

MODEL_PATH="/data/oceanus_share/shangshouduo-jk/project/ckpts/Qwen/qwen/Qwen3-VL-8B-Instruct" 
INFERENCE_SCRIPT="/data/oceanus_share/shangshouduo-jk/myproject/src/models/run.py"

# q-frame 离线处理输出的 manifest（包含 image_paths）
QFRAME_MANIFEST="/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MLVU/baselines/q-frame/selected_frames_manifest.jsonl"

N_PROC=1
OUTPUT_DIR="/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip"
DATASET_NAME="MLVU"
GPU_MEMORY=0.5

# 显存/临时目录优化（按需调整）
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=20480
export TMPDIR=/data/oceanus_share/shangshouduo-jk/tmp
export TEMP=/data/oceanus_share/shangshouduo-jk/tmp
export TMP=/data/oceanus_share/shangshouduo-jk/tmp
mkdir -p /data/oceanus_share/shangshouduo-jk/tmp

set -e

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
  exit 1
fi

if [ ! -f "$QFRAME_MANIFEST" ]; then
  echo "ERROR: QFRAME_MANIFEST not found: $QFRAME_MANIFEST"
  exit 1
fi

RETRY_COUNT=0
while true; do
  echo "---------------------------------------------------------" | tee -a "$LOG_FILE"
  echo "Starting Inference (Attempt #$((RETRY_COUNT+1))) at $(date)" | tee -a "$LOG_FILE"
  echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
  echo "Dataset: $DATASET_NAME" | tee -a "$LOG_FILE"
  echo "Manifest: $QFRAME_MANIFEST" | tee -a "$LOG_FILE"
  echo "---------------------------------------------------------" | tee -a "$LOG_FILE"

  set +e
  CUDA_VISIBLE_DEVICES=2,3 python "$INFERENCE_SCRIPT" \
    --nproc "$N_PROC" \
    --model_name "Qwen3-VL-8B-Instruct" \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET_NAME" \
    --output "$OUTPUT_DIR" \
    --gpu_memory_utilization "$GPU_MEMORY" \
    --sample_path "$QFRAME_MANIFEST" \
    --use_vllm \
    2>&1 | tee -a "$LOG_FILE"
  EXIT_CODE=$?
  set -e

  if [ $EXIT_CODE -eq 0 ]; then
    echo "Inference finished successfully at $(date)!" | tee -a "$LOG_FILE"
    break
  else
    echo "Process crashed with exit code $EXIT_CODE at $(date)." | tee -a "$LOG_FILE"
    echo "Waiting 120s before restart..." | tee -a "$LOG_FILE"
    sleep 120
    RETRY_COUNT=$((RETRY_COUNT+1))
  fi
done

echo "Full log saved to: $LOG_FILE"

