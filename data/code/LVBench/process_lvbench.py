import json
import os

# 定义文件路径
input_path = '/data/oceanus_share/shangshouduo-jk/myproject/data/processed/LVBench/video_info.meta_missing.jsonl'
output_path = '/data/oceanus_share/shangshouduo-jk/myproject/data/processed/LVBench/video_info.meta_missing.json'
# 定义视频存放的基础路径
base_video_dir = '/data/oceanus_share/shangshouduo-jk/project/datasets/LVBench/videos'

def process_data(input_file, output_file):
    flattened_data = []
    
    print(f"正在读取文件: {input_file} ...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    video_entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行格式错误，已跳过。")
                    continue
                
                # 提取公共信息
                key = video_entry.get('key', '')
                video_type = video_entry.get('type', '')
                video_info = video_entry.get('video_info', {})
                
                # 生成 video_path
                # 使用 os.path.join 确保路径拼接正确，手动拼接确保符合你的具体要求
                # 你的要求：路径 + key + .mp4
                video_path = os.path.join(base_video_dir, f"{key}.mp4")
                
                # 遍历 qa 列表，拆解成独立的数据条目
                qa_list = video_entry.get('qa', [])
                for qa_item in qa_list:
                    # 构建新的数据条目
                    new_item = {
                        "key": key,
                        "video_path": video_path,
                        "type": video_type,
                        "video_info": video_info,
                        # 将 qa_item 中的字段（uid, question, answer等）合并进来
                        **qa_item
                    }
                    flattened_data.append(new_item)
        
        # 写入最终的 json 文件
        print(f"处理完成，共生成 {len(flattened_data)} 条数据。")
        print(f"正在写入文件: {output_file} ...")
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(flattened_data, out_f, indent=4, ensure_ascii=False)
            
        print("写入成功！")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    process_data(input_path, output_path)