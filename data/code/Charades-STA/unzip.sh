#!/bin/bash

# 定义数据集根目录
DATA_DIR="/data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA"

# 进入该目录
cd "$DATA_DIR" || { echo "Directory not found!"; exit 1; }

# 创建 video 文件夹
mkdir -p video

# 解压四个压缩包到 video/ 目录
echo "Unzipping part 1..."
unzip -o -j "Charades_v1_480_part_1.zip" -d video/

echo "Unzipping part 2..."
unzip -o -j "Charades_v1_480_part_2.zip" -d video/

echo "Unzipping part 3..."
unzip -o -j "Charades_v1_480_part_3.zip" -d video/

echo "Unzipping part 4..."
unzip -o -j "Charades_v1_480_part_4.zip" -d video/

echo "All videos extracted to $DATA_DIR/video/"