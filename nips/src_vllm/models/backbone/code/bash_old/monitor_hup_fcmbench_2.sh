#!/bin/bash
# chmod +x run.sh

# --- 关键设置：确保管道命令(|)也能捕获到 Python 的错误 ---
set -o pipefail 

# --- 自动日志设置 ---
LOG_DIR="/data/oceanus_share/shangshouduo-jk/myproject/output/logs/fcmbench"
mkdir -p "$LOG_DIR"

# 注意：日志文件名只在脚本开始时生成一次，重启时会追加到同一个文件
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Log file will be: $LOG_FILE"
echo "========================================================="

# 1. 模型权重路径
MODEL_PATH="/data/oceanus_share/cuirunze-jk/ckpts/Qwen/Qwen3-VL-8B-Instruct" 

# 2. 使用的进程数量
N_PROC=1

# 3. 结果输出目录
OUTPUT_DIR="/data/oceanus_share/shangshouduo-jk/myproject/output/results"

# 4. 数据集名称
# DATASET_NAME="Video-MME"
# DATASET_NAME="MLVU"
# DATASET_NAME="LVBench"
# DATASET_NAME="MMVU"
# DATASET_NAME="MVBench"
DATASET_NAME="FCMBench"

GPU_MEMORY=0.7

# 5. Python 脚本的文件名
INFERENCE_SCRIPT="/data/oceanus_share/shangshouduo-jk/myproject/src/models/run.py"

# 显存优化参数
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=20480
export TMPDIR=/data/oceanus_share/shangshouduo-jk/tmp
export TEMP=/data/oceanus_share/shangshouduo-jk/tmp
export TMP=/data/oceanus_share/shangshouduo-jk/tmp
mkdir -p /data/oceanus_share/shangshouduo-jk/tmp

set -e 

echo "========================================================="
echo "Starting Job at $(date)"
echo "Model Path: $MODEL_PATH"
echo "GPUs (N_PROC): $N_PROC"
echo "========================================================="

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ Warning: Model path '$MODEL_PATH' does not exist or is not a directory."
    exit 1
fi

# ========================================================
# 🔁 自动重启循环逻辑 (Auto-Restart Loop)
# ========================================================

RETRY_COUNT=0

while true; do
    echo "---------------------------------------------------------"
    echo "🚀 Starting Inference (Attempt #$((RETRY_COUNT+1)))..."
    echo "---------------------------------------------------------"

    echo "---------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Starting Inference (Attempt #$((RETRY_COUNT+1))) at $(date)" | tee -a "$LOG_FILE"
    echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "Dataset: $DATASET_NAME" | tee -a "$LOG_FILE"
    echo "Manifest: $AKS_SAMPLE" | tee -a "$LOG_FILE"
    echo "---------------------------------------------------------" | tee -a "$LOG_FILE"


    # 运行 Python 脚本
    # 使用 tee -a (append) 模式，避免覆盖之前的日志
    # set +e 暂时允许错误，以便我们在脚本里处理它，而不是直接退出 shell
    set +e 
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python "$INFERENCE_SCRIPT" \
        --nproc "$N_PROC" \
        --model_name "Qwen3-VL-8B-Instruct" \
        --model_path "$MODEL_PATH" \
        --dataset "$DATASET_NAME" \
        --output "$OUTPUT_DIR" \
        --gpu_memory_utilization "$GPU_MEMORY" \
        --use_vllm \
        2>&1 | tee -a "$LOG_FILE"

    # 获取上一个命令（Python）的退出码
    EXIT_CODE=$?
    set -e # 恢复严格模式

    # --- 判断逻辑 ---
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Inference finished successfully at $(date)!"
        break  # 成功跑完，跳出循环
    else
        echo "❌ Process crashed with exit code $EXIT_CODE at $(date)."
        echo "⚠️  Likely OOM or Runtime Error. "
        echo "⏳ Waiting for 2 minutes (120s) to cool down GPU and clear memory..."
        
        # 这里就是你要求的“每隔两分钟检测”的变体：
        # 既然挂了，我们等两分钟再重启，给显卡喘息时间
        sleep 120 
        
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "🔄 Restarting process now..."
    fi
done

echo "📄 Full log saved to: $LOG_FILE"