import os
import json
import glob

# 配置路径
input_dir = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MVBench/json_new_change"
output_file = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MVBench/mvbench_all.json"

# 获取所有 JSON 文件（按名称排序以保证可重复性）
json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

print(f"🔍 找到 {len(json_files)} 个 JSON 文件，开始合并...")

all_data = []

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                print(f"⚠️ 警告：{file_path} 不是列表格式，跳过。")
        except Exception as e:
            print(f"❌ 错误：无法读取 {file_path} - {e}")

# 写入合并后的文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)

print(f"✅ 合并完成！共 {len(all_data)} 条数据，已保存至：{output_file}")