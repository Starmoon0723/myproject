import json
import os

# 定义文件路径
# json_file_path = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench/json_new_change/fine_grained_pose.json"
# base_video_dir = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench/video/unpacked/rgb"

json_file_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MVBench/json_new_change/episodic_reasoning.json"
base_video_dir = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench/video/unpacked/tvqa/frames_fps3_hq"

# 读取 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 更新每个条目的 video_path
for item in data:
    video_name = item["video"]
    item["video_path"] = os.path.join(base_video_dir, video_name) + ".mp4"

# 写回原文件（覆盖）
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ Successfully updated 'video_path' for {len(data)} entries in {json_file_path}")