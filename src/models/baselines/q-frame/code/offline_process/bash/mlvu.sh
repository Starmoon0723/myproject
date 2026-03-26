#!/bin/bash

PROJECT_ROOT="/data/oceanus_share/shangshouduo-jk/myproject"
CODE_PATH="$PROJECT_ROOT/src/models/baselines/q-frame/code/offline_process/offline_videomme_qframe.py"
LONGCLIP_ROOT="$PROJECT_ROOT/src/models/baselines/q-frame/Long-CLIP"
LONGCLIP_CKPT="/data/oceanus_share/shangshouduo-jk/myproject/ckpts/longclip/longclip-L.pt" # 默认加载 Long-CLIP 权重名

# 数据集配置
# DATASET_NAME="Video-MME"
# DATASET="LVBench"
DATASET="MLVU"
# DATASET="VideoMMMU"
# DATASET="MMVU"
# DATASET="MVBench"

if [ "$DATASET" == "Video-MME" ]; then
    INPUT_PATH="/data/oceanus_share/shangshouduo-jk/project/datasets/Video-MME/Video-MME_only_question.tsv"
    OUTPUT_DIR="$PROJECT_ROOT/data/processed/Video-MME/baselines/q-frame"
elif [ "$DATASET" == "LVBench" ]; then
    INPUT_PATH="/data/oceanus_share/shangshouduo-jk/myproject/data/processed/LVBench/video_info.meta_wo_options.json"
    OUTPUT_DIR="$PROJECT_ROOT/data/processed/LVBench/baselines/q-frame"
elif [ "$DATASET" == "MLVU" ]; then
    INPUT_PATH="/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MLVU/mlvu_all.json"
    OUTPUT_DIR="$PROJECT_ROOT/data/processed/MLVU/baselines/q-frame"
elif [ "$DATASET" == "VideoMMMU" ]; then
    INPUT_PATH="/data/oceanus_share/shangshouduo-jk/myproject/data/processed/VideoMMMU/VideoMMMU.tsv"
    OUTPUT_DIR="$PROJECT_ROOT/data/processed/VideoMMMU/baselines/q-frame"
elif [ "$DATASET" == "MMVU" ]; then
    INPUT_PATH="/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MMVU/validation_process.json"
    OUTPUT_DIR="$PROJECT_ROOT/data/processed/MMVU/baselines/q-frame"
elif [ "$DATASET" == "MVBench" ]; then
    INPUT_PATH="/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MVBench/mvbench_all.json"
    OUTPUT_DIR="$PROJECT_ROOT/data/processed/MVBench/baselines/q-frame"
fi



# ==============================================================================
# q-frame 核心超参数
# ==============================================================================
MAX_FRAMES=128
FPS=1.0
HIGH_F=4
MID_F=8
LOW_F=32
TAU=0.8

# ==============================================================================
# 运行指令
# ==============================================================================

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------------------------------"
echo "Starting q-frame offline preprocessing..."
echo "Dataset: $DATASET_NAME"
echo "TSV: $INPUT_PATH"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------------------------------"

# 执行 Python 脚本
# --resume: 断点续传
# --image_ext: 建议复现时使用 png 以保证无损，或 jpg (quality 95) 节省空间
CUDA_VISIBLE_DEVICES="2" python "$CODE_PATH" \
    --tsv_path "$INPUT_PATH" \
    --output_root "$OUTPUT_DIR" \
    --longclip_root "$LONGCLIP_ROOT" \
    --longclip_ckpt "$LONGCLIP_CKPT" \
    --max_frames_num $MAX_FRAMES \
    --target_fps $FPS \
    --high_frames $HIGH_F \
    --mid_frames $MID_F \
    --low_frames $LOW_F \
    --tau $TAU \
    --image_ext "jpg" \
    --resume 
