import json
import os

# 定义文件路径和基础路径
input_file_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/FCMBench/fcmbench_longvideo_v1.0_20260205.jsonl"
output_file_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/FCMBench/fcmbench_longvideo_v1.0_20260205_absolute.jsonl"
base_video_dir = "/data/oceanus_share/shangshouduo-jk/project/datasets/FCMBench/video"

def process_jsonl():
    print(f"正在处理文件: {input_file_path}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
            
            count = 0
            for line in f_in:
                if not line.strip():
                    continue
                
                # 1. 读取 JSON
                data = json.loads(line)
                
                # 2. 修改 video_path
                if "video_path" in data:
                    original_path = data["video_path"]
                    # 使用 os.path.join 智能拼接路径，防止斜杠重复或缺失
                    new_path = os.path.join(base_video_dir, original_path)
                    data["video_path"] = new_path
                
                # 3. 写入新文件
                # ensure_ascii=False 是关键，它确保中文不会被转义成 \uXXXX 格式
                json_line = json.dumps(data, ensure_ascii=False)
                f_out.write(json_line + '\n')
                count += 1
                
        print(f"处理完成！")
        print(f"共处理了 {count} 行数据。")
        print(f"新文件已保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    process_jsonl()