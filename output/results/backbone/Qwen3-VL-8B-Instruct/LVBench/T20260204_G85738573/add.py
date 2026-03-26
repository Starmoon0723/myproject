import json

input_path = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/backbone/Qwen3-VL-8B-Instruct/LVBench/T20260204_G85738573/Qwen3-VL-8B-Instruct_LVBench.jsonl"
output_path = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/backbone/Qwen3-VL-8B-Instruct/LVBench/T20260204_G85738573/Qwen3-VL-8B-Instruct_LVBench_reindexed.jsonl"

start_index = 1492

with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:
    for i, line in enumerate(fin):
        if not line.strip():
            continue  # 跳过空行
        data = json.loads(line.strip())
        data["index"] = start_index + i
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Re-indexed from {start_index} to {start_index + i}, saved to {output_path}")