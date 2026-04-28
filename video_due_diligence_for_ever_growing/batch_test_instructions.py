import json
import base64
import os
from openai import OpenAI
from json_repair import repair_json

# --- 配置区 ---
client = OpenAI(api_key="Bearer sk-8ef8cc0e93ec5a04bb1892ce1bf97148", base_url='https://litellm-manage-dev.sandbox.deepbank.daikuan.qihoo.net/v1')

INPUT_FILE = "fcmbench_longvideo_v1.0_20260228.jsonl"
OUTPUT_FILE = "test_cn.jsonl"
VIDEO_PREFIX = r"D:\视频拼接"
MODEL_NAME = "deepbank/qwen3-vl-32b" 

def encode_video_to_base64(video_path):
    """将视频文件编码为 base64 字符串"""
    if not os.path.exists(video_path):
        print(f"Warning: File not found {video_path}")
        return None
    with open(video_path, "rb") as video_file:
        b64 = base64.b64encode(video_file.read()).decode('utf-8')
    return f"data:video/mp4;base64,{b64}"


def process_jsonl():
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line)
            prompt_text = data.get("prompt", "")
            # 拼接固定前缀路径
            raw_video_path = data.get("video_path", "")
            full_video_path = os.path.normpath(
                os.path.join(VIDEO_PREFIX, raw_video_path.lstrip("/\\"))
            )
            extra_body = {
                "mm_processor_kwargs": {
                    "fps": 2,
                    "do_sample_frames": True
                }
            }
            
            # 编码视频
            base64_video = encode_video_to_base64(full_video_path)
            
            if base64_video:
                try:
                    # 调用 OpenAI SDK
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {
                                        "type": "video_url", # 注意：不同模型厂商对视频字段的定义可能略有不同
                                        "video_url": {"url": base64_video }
                                    }
                                ],
                            }
                        ],
                        temperature=0.1,
                        extra_body = extra_body
                    )
                    
                    # 获取模型返回结果
                    model_output = response.choices[0].message.content
                    clear_json = repair_json(model_output, return_objects=True)

                    # 将结果拼接到原始数据字段中（例如增加 model_response 键）
                    data["model_response"] = clear_json
                    
                except Exception as e:
                    print(f"Error processing video {full_video_path}: {e}")
                    data["model_response"] = f"Error: {str(e)}"
            else:
                data["model_response"] = "Error: Video file not found or empty"

            # 写入输出文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            outfile.flush()

if __name__ == "__main__":
    process_jsonl()