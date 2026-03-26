import json
import os

# 定义文件路径
# input_path = "/data/oceanus_share/shangshouduo-jk/project/code/frame_extraction/dataset/method/aks/videomme/selected_frames_offical/videomme/clip/videomme_clip_64.json"
# output_path = os.path.join(os.path.dirname(input_path), "videomme_clip_64.jsonl")

# input_path = "/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/selected_frames/videomme/clip/selected_frames.json"
input_path = "/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/selected_frames/mlvu/clip/selected_frames.json"
# ”/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/selected_frames/videomme“
output_path = os.path.join(os.path.dirname(input_path), "selected_frames.jsonl")
def convert_to_jsonl(in_file, out_file):
    # 1. 读取原始 JSON 数据
    try:
        with open(in_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"成功读取文件，共包含 {len(data)} 条数据列表。")
    except FileNotFoundError:
        print(f"错误：找不到文件 {in_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {in_file} 不是合法的 JSON 格式")
        return

    # 2. 转换并写入 JSONL
    with open(out_file, 'w', encoding='utf-8') as f_out:
        for sub_list in data:
            # 这里的逻辑是：
            # 1. 遍历外层列表的每一个子列表
            # 2. 将子列表中的 float (如 0.0) 转换为 int (如 0)
            # 3. 构造目标字典格式
            
            # 确保转换为整数以匹配你的目标格式
            int_indices = [int(frame) for frame in sub_list]
            
            record = {
                "frame_indices": int_indices
            }
            
            # 写入一行，使用 ensure_ascii=False 防止中文乱码（虽然这里全是数字）
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"转换完成！文件已保存至：\n{out_file}")

if __name__ == "__main__":
    convert_to_jsonl(input_path, output_path)