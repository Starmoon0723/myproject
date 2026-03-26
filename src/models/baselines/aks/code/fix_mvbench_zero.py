import json

file_path = "/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/selected_frames/mvbench/clip/selected_frames.jsonl"

# 读取并修改
modified_lines = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        data = json.loads(line)
        if data.get("frame_indices") == []:
            data["frame_indices"] = [0]
        modified_lines.append(json.dumps(data))

# 写回原文件
with open(file_path, 'w', encoding='utf-8') as f:
    for line in modified_lines:
        f.write(line + '\n')