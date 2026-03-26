import base64
from openai import OpenAI


# 初始化客户端（替换为你自己的 endpoint 和 API key）
client = OpenAI(
    base_url="http://127.0.0.1:8901/v1",  # 替换为你的实际 endpoint
    api_key="EMPTY"              # 替换为你的实际 token
)


# 读取视频并转为 base64（假设视频是 mp4）
with open("/data/oceanus_share/cuirunze-jk/video/视频拼接3/35-尉迟琛/35-尉迟琛_20s_L2_1.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode("utf-8")


# 构造消息内容
messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请分析视频，根据预定义的文件类型，列出视频中出现的文件。以Python列表格式输出。\n预定义列表：[\"不动产权证书\",\"不动产信息查询结果\",\"贷款结清证明\",\"贷款申请表\",\"个人贷款调查报告\",\"个人所得税完税证明\",\"户口本（本人页）\",\"户口本（户主页）\",\"结婚证（电子版）\",\"离婚证（电子版）\",\"社会保险参保证明\",\"身份证（国徽面）\",\"身份证（人像面）\",\"收入证明\",\"银行卡\",\"资金流水分析表\",\"企业法人营业执照\",\"食品经营许可证\",\"流水明细\",\"交易记录\"]。\n示例输出：{\"answer\": [\"贷款结清证明\", \"不动产权证书\", \"户口本（户主页）\", \"户口本（本人页）\"]}。\n。"}, #换成prompt
            {
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{video_b64}"
                }
            }
        ]
    }
]



response = client.chat.completions.create(
    model="/data/oceanus_share/cuirunze-jk/ckpts/Qwen/Qwen3-VL-32B-Instruct", #换成完整模型路径
    messages=messages,
    temperature=0.1,
            # timeout=args.timeout,
    extra_body = {
    "mm_processor_kwargs": {
    "fps": 4,
    "do_sample_frames": True
    }
    }
)


print(response)