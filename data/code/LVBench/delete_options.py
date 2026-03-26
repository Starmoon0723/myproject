import json

input_file = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/LVBench/video_info.meta.json"
output_file = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/LVBench/video_info.meta_wo_options.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    question = item["question"]
    # 以 "\n(A)" 为分界，取前面的部分
    if "\n(A)" in question:
        item["question"] = question.split("\n(A)")[0].strip()
    else:
        item["question"] = question.strip()  # 如果没有选项，保留原样

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)