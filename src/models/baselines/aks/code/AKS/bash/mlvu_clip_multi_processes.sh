#!/bin/bash

# PROJECT_ROOT="/data/oceanus_share/shangshouduo-jk/project/github/AKS"
MODEL_CACHE_DIR="/data/oceanus_share/shangshouduo-jk/myproject/ckpts"

cd "$PROJECT_ROOT" || { echo "Project directory not found!"; exit 1; }

# 设置模型缓存路径（关键！）
export HF_HOME="$MODEL_CACHE_DIR"
export LAVIS_ROOT="$MODEL_CACHE_DIR"          # LAVIS 会使用这个作为根目录
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export TORCH_HOME="$MODEL_CACHE_DIR"          # 部分 torch 模型也会用到

# 确保目录存在
mkdir -p "$MODEL_CACHE_DIR"

echo "📦 Model cache directory: $MODEL_CACHE_DIR"
echo "🚀 Starting feature extraction..."

# videomme, mvbench, lvbench, mlvu, mmvu, videommmu

CUDA_VISIBLE_DEVICES=0,1 python /data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/feature_extract_new_multi_processes.py \
    --dataset_name "mlvu" \
    --label_path "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MLVU" \
    --video_path "/data/oceanus_share/shangshouduo-jk/project/datasets/MVLU/MLVU/video" \
    --extract_feature_model "clip" \
    --output_file "/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/outscores" \
    --blip_model_path "/data/oceanus_share/shangshouduo-jk/myproject/ckpts" \
    --clip_model_path "/data/oceanus_share/shangshouduo-jk/myproject/ckpts" \
    --device cuda \
    --gpu_ids 0,1 \
    --workers_per_gpu 4 \
    --num_workers 8 \
    --chunk_size 32