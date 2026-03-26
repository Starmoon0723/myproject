import json
import logging
import os.path as osp

import numpy as np
import pandas as pd
from pathlib import Path

from .base import VideoDataset

logger = logging.getLogger(__name__)


# class MMVUDataset(VideoDataset):
#     def __init__(self, processor, **kwargs):
#         data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MMVU"
#         # data_path = osp.join(data_root, "mmvu_test.json")
#         data_path = osp.join(data_root, "validation_process.json")

#         with open(data_path, "r", encoding="utf-8") as f:
#             data_list = json.load(f)

#         # 转成 DataFrame
#         df = pd.DataFrame(data_list)
#         if "index" not in df.columns:
#             df["index"] = np.arange(len(df))

#         super().__init__(
#             processor,
#             dataset_name="MMVU",
#             data_root=data_root,
#             data_path=None,
#             dataframe=df,
#             **kwargs,
#         )

#     def _build_struct(self, line):
#         # 视频路径
#         video_path = osp.join(self.data_root, "videos", line["video"])

#         # question
#         question = line["question"]
#         q_type = line.get("question_type", "multiple-choice")
#         choices = line.get("choices", {})

#         # === 构建 prompt ===
#         if q_type == "multiple-choice":
#             prompt_text = (
#                 f"Question: {question}\n"
#                 f"A: {choices['A']}\n"
#                 f"B: {choices['B']}\n"
#                 f"C: {choices['C']}\n"
#                 f"D: {choices['D']}\n"
#                 f"E: {choices.get('E', '')}\n"
#                 "Visual Information: processed video\n"
#                 "Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices."
#             )
#             # 移除空的选项 (如果E不存在)
#             prompt_text = prompt_text.replace("E: \n", "")

#         else:
#             # Open-ended
#             prompt_text = (
#                 f"Question: {question}\n"
#                 "Visual Information: processed video\n"
#                 "Do not generate any intermediate reasoning process. Directly output the final answer."
#             )

#         struct = [
#             {"type": "video", "value": video_path},
#             {"type": "text", "value": prompt_text},
#         ]
#         return struct

#     @classmethod
#     def evaluate(cls, eval_file, **judge_kwargs):
#         # 这里你可以实现简单的规则评测，或者留空
#         logger.info("MMVU evaluation requires Hybrid Judge (Rule + GPT). Please run external eval script.")
#         return None


class MMVUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        self.use_cot = True

        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MMVU"
        json_path = osp.join(data_root, "validation_process.json") 
        
        logger.info(f"Loading MMVU data from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            
        # 2. 转换为 DataFrame 格式，这是你的基类需要的
        # 确保包含 index, video_path 等字段
        all_rows = []
        for idx, item in enumerate(data_list):
            row = {
                "index": idx,
                "question_id": item.get("id", idx),
                "video_path": item['video_path'],
                "question_type": item["question_type"], # 'multiple-choice' or 'open-ended'
                "question": item["question"],
                "choices": item.get("choices", None), # 只有选择题有
                "answer": item["answer"],
                "category": Path(item['video_path']).parent.name
            }
            all_rows.append(row)

        df = pd.DataFrame(all_rows)
        logger.info(f"Loaded MMVU dataset: {len(df)} samples")

        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="MMVU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            sample_path=sample_path,
            # fps=2, # MMVU 默认配置
            **kwargs
        )

    def _build_struct(self, line):
        struct = []
        # video_path = line["video_path"]

        image_paths = line.get("image_paths", None)
        if isinstance(image_paths, str):
            try:
                image_paths = json.loads(image_paths)
            except Exception:
                image_paths = None

        if isinstance(image_paths, list) and len(image_paths) > 0:
            # q-frame: multi-image input
            for p in image_paths:
                struct.append(dict(type="image", value=str(p)))
        else:
            fi = line.get("frame_indices", None)
            v = dict(type="video", value=line["video_path"])
            if fi is not None:
                v["frame_indices"] = fi
            struct.append(v)

        q_type = line["question_type"]
        question = line["question"]
        
        # --- 复刻 lmms-eval 的 Prompt 逻辑 ---
        if q_type == "multiple-choice":
            choices = line["choices"]
            # 注意：lmms-eval 的 prompt 比较简单
            if self.use_cot:
                prompt_text = (
                    f"Question: {question}\n"
                    f"A: {choices['A']}\n"
                    f"B: {choices['B']}\n"
                    f"C: {choices['C']}\n"
                    f"D: {choices['D']}\n"
                    f"E: {choices.get('E', '')}\n"  # MMVU 有些有E选项
                    "Visual Information: processed video\n"
                    "Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: \"Therefore, the final answer is: $LETTER\" (without quotes), where $LETTER is one of the options. Think step by step before answering."
                )
            else:
                prompt_text = (
                    f"Question: {question}\n"
                    f"A: {choices['A']}\n"
                    f"B: {choices['B']}\n"
                    f"C: {choices['C']}\n"
                    f"D: {choices['D']}\n"
                    f"E: {choices.get('E', '')}\n" # MMVU 有些有E选项
                    "Visual Information: processed video\n"
                    "Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices."
                )
            # 移除空的选项 (如果E不存在)
            prompt_text = prompt_text.replace("E: \n", "")
            
        else:
            if self.use_cot:
                # Open-ended
                prompt_text = (
                    f"\nQuestion: {question}\n"
                    "Visual Information: processed video\n"
                    "Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: \"Therefore, the final answer is: Answer: $ANSWER\" (without quotes), where $ANSWER is the final answer of the question. Think step by step before answering."
                )
            else:
                # Open-ended
                prompt_text = (
                    f"Question: {question}\n"
                    "Visual Information: processed video\n"
                    "Do not generate any intermediate reasoning process. Directly output the final answer."
                )

        # struct = [
        #     {"type": "video", "value": video_path},
        #     {"type": "text", "value": prompt_text},
        # ]
        struct.append(dict(type="text", value=prompt_text))
        
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger.info("MMVU evaluation requires Hybrid Judge (Rule + GPT). Please run external eval script.")
        return None
