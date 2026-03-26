#!/bin/bash

DATASET_NAME="mvbench"
EXTRACT_FEATURE_MODEL="clip"
SCORE_PATH="/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/outscores/mvbench/clip/scores.json"
FRAME_PATH="/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/outscores/mvbench/clip/frames.json"
MAX_NUM_FRAMES=64
RATIO=1
T1=0.8
T2=-100
ALL_DEPTH=5
OUTPUT_FILE="/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/selected_frames"

# 运行 Python 脚本
python /data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/frame_select.py \
    --dataset_name "$DATASET_NAME" \
    --extract_feature_model "$EXTRACT_FEATURE_MODEL" \
    --score_path "$SCORE_PATH" \
    --frame_path "$FRAME_PATH" \
    --max_num_frames "$MAX_NUM_FRAMES" \
    --ratio "$RATIO" \
    --t1 "$T1" \
    --t2 "$T2" \
    --all_depth "$ALL_DEPTH" \
    --output_file "$OUTPUT_FILE"