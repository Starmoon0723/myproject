import os
from openai import OpenAI


# ====== 配置区 ======
BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = "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"
MODEL = "minimaxai/minimax-m2.5"


def chat_once(prompt: str) -> str:
    """非流式：一次性拿到完整回复"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def chat_stream(prompt: str) -> str:
    """流式：边生成边打印，同时汇总返回"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        stream=True,
    )

    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            chunks.append(delta.content)

    print()  # 换行
    return "".join(chunks)


def main():
    prompt = "你是谁？"

    # print("=== Non-stream ===")
    # try:
    #     out = chat_once(prompt)
    #     print(out)
    # except Exception as e:
    #     print(f"[ERROR] non-stream failed: {e}")

    print("\n=== Stream ===")
    try:
        _ = chat_stream(prompt)
    except Exception as e:
        print(f"[ERROR] stream failed: {e}")


if __name__ == "__main__":
    main()