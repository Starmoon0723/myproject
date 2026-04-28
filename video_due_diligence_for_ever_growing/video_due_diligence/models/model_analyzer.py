import os
import json
import base64
import time
import asyncio
import yaml
from openai import AsyncOpenAI
from json_repair import repair_json
import pprint


class ModelAnalyzer:
    def __init__(self, config_file=None):
        # 1. 自动定位并加载配置文件
        if config_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, "..", "..", "config.yaml")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        vllm_cfg = self.config.get("vllm", {})

        # --- 视频配置 (Video) ---
        video_cfg = vllm_cfg.get("video", {})
        
        # 安全读取策略：配置文件 -> 环境变量 -> 默认值
        v_api_key = video_cfg.get("api_key") or os.getenv("VIDEO_API_KEY", "EMPTY")
        v_base_url = video_cfg.get("base_url") or os.getenv("VIDEO_BASE_URL")
        
        self.video_client = AsyncOpenAI(api_key=v_api_key, base_url=v_base_url)
        self.video_model = video_cfg.get("model_name")
        self.video_system_prompt = video_cfg.get("system_prompt", "你是一个视频分析专家。")
        self.video_prompt_tpl = video_cfg.get("prompt", "{task_list}")
        self.video_sem = asyncio.Semaphore(video_cfg.get("max_concurrent", 3))

        # --- ASR 配置 (ASR) ---
        asr_cfg = vllm_cfg.get("asr", {})
        
        # 安全读取策略：配置文件 -> 环境变量 -> 默认值
        a_api_key = asr_cfg.get("api_key") or os.getenv("ASR_API_KEY", "EMPTY")
        a_base_url = asr_cfg.get("base_url") or os.getenv("ASR_BASE_URL")
        
        self.asr_client = AsyncOpenAI(api_key=a_api_key, base_url=a_base_url)
        self.asr_model = asr_cfg.get("model_name")
        self.asr_sem = asyncio.Semaphore(asr_cfg.get("max_concurrent", 10))

    def _ensure_base64(self, input_data, mime_type=None):
        """核心兼容逻辑：视频带前缀，音频不带前缀"""
        raw_b64 = ""
        if isinstance(input_data, str) and os.path.exists(input_data):
            with open(input_data, "rb") as f:
                raw_b64 = base64.b64encode(f.read()).decode('utf-8')
        else:
            raw_b64 = input_data.split(",")[-1] if isinstance(input_data, str) and "," in input_data else input_data

        return f"data:{mime_type};base64,{raw_b64}" if mime_type else raw_b64

    async def analyze_asr(self, request_id, audio_input, stream=False):
        """异步音频分析 (ASR)"""
        async with self.asr_sem:
            # ASR 通常只需要纯 base64 字符串，不需要 mime 前缀
            audio_b64 = self._ensure_base64(audio_input)
            
            messages = [{
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}]
            }]
            
            # 调用异步模型接口
            content = await self._call_model(self.asr_client, self.asr_model, messages, stream=stream)
            
            # 提取识别文本
            response_text = content.split("<asr_text>")[-1].strip() if "<asr_text>" in content else content
            
            # 返回你要求的固定数据结构
            result = {
                "result_type": "asr",
                "request_id": request_id,
                "status": "SUCCESS" if "API_ERROR" not in content else content ,
                "analysis": response_text
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

    async def analyze_video(self, request_id, video_input, task_list=None, stream=False):
        """异步视频分析"""
        async with self.video_sem:
            video_url = self._ensure_base64(video_input, mime_type="video/mp4")
            prompt_text = self.video_prompt_tpl.format(task_list=task_list)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.video_system_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "video_url", "video_url": {"url": video_url}}
                ]}
            ]

            content = await self._call_model(self.video_client, self.video_model, messages, stream=stream)
            
            try:
                response_obj = repair_json(content, return_objects=True)
                # 2. 提取并清洗命中的项
                hit_analysis = self._filter_hit_items(response_obj)
            except:
                hit_analysis = content

            result = {
                "result_type": "video",
                "request_id": request_id,
                "status": "SUCCESS" if "API_ERROR" not in content else content,
                "analysis": hit_analysis
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

    async def _call_model(self, client, model, messages, stream=False):
        """通用异步请求封装"""
        try:
            if not stream:
                response = await client.chat.completions.create(
                    model=model, messages=messages, temperature=0.1, stream=False
                )
                return response.choices[0].message.content
            else:
                resp = await client.chat.completions.create(
                    model=model, messages=messages, temperature=0.1, stream=True
                )
                full = []
                async for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full.append(chunk.choices[0].delta.content)
                return "".join(full)
        except Exception as e:
            return f"API_ERROR: {str(e)}"
    

    def _filter_hit_items(self, data):
        """
        针对嵌套结构进行过滤：
        如果 '清晰度', '完整性', '详情' 中有任意一个是 '未命中'，则剔除该项。
        """
        if not isinstance(data, dict):
            return data
        
        # 1. 自动剥离最外层的 'answer' 壳子
        target = data.get("answer", data)
        
        if not isinstance(target, dict):
            return target

        filtered_dict = {}
        
        # 定义需要检查的字段以及“未命中”的判定标准
        required_keys = ["清晰度", "完整性", "详情"]
        miss_flags = ["未命中", "未命中。", None, "null", ""]

        for item_name, details in target.items():
            # 如果 details 不是字典（模型偶尔抽风返回字符串），直接跳过或按需处理
            if not isinstance(details, dict):
                continue
                
            # --- 核心判定逻辑 ---
            # 提取这三个字段的值
            v_clarity = details.get("清晰度")
            v_completeness = details.get("完整性")
            v_detail = details.get("详情")
            
            # 逻辑判断：三个字段必须全部都不在 miss_flags 中
            # 换句话说：只要有一个命中 miss_flags，is_hit 就是 False
            is_hit = all(
                v not in miss_flags for v in [v_clarity, v_completeness, v_detail]
            )
            
            if is_hit:
                filtered_dict[item_name] = details
                
        return filtered_dict        

async def test_mixed_concurrency():
    # 1. 初始化
    analyzer = ModelAnalyzer()
    
    # 2. 准备纯数据（不是协程）
    video_paths = ['IMG_2663_000_4fps.mp4'] * 4
    # audio_paths = ["test.wav"] * 6
    task_list = ['企业大门门头', '生产车间', '品类', '存量规模', '生产机器设备', '运行状态','产能', '生产质量', '现场作业人员', '出勤班表']

    # 3. 在这里统一创建协程任务列表
    tasks = []
    
    # 添加 6 个视频任务
    for i, path in enumerate(video_paths):
        # 仅仅是把协程加入列表，并没有 await 它
        tasks.append(analyzer.analyze_video(f"vid_ID_{i}", path, task_list=task_list))
        
    # # 添加 6 个音频任务
    # for i, path in enumerate(audio_paths):
    #     tasks.append(analyzer.analyze_asr(f"asr_ID_{i}", path))
    
    start_time = time.perf_counter()
    print(f">>> [0.00s] 开始混合并发测试 (6个视频 + 6个音频)...")

    # 4. 执行并观察
    # as_completed 会立即开始同时执行 tasks 列表里的所有协程
    for future in asyncio.as_completed(tasks):
        try:
            result_str = await future
            result = json.loads(result_str)
            
            elapsed = time.perf_counter() - start_time
            rid = result.get("request_id")
            rtype = result.get("result_type")
            status = result.get("status")
            
            print(f"[{elapsed:.2f}s] 收到响应 | 类型: {rtype:5} | ID: {rid:10} | 状态: {status}")
            
            # content 摘要
            content = result.get("analysis")
            pprint.pprint(result)
            
        except Exception as e:
            print(f"任务执行出错: {e}")

    total_time = time.perf_counter() - start_time
    print(f"\n>>> 所有任务处理完成，总耗时: {total_time:.2f}s")

if __name__ == "__main__":
    # 执行混合并发测试
    try:
        asyncio.run(test_mixed_concurrency())
    except KeyboardInterrupt:
        pass