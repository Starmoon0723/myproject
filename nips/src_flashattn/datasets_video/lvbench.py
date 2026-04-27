import glob
import json
import logging
import os.path as osp
import re

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class LVBenchDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/LVBench/video_info.meta.json"
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/LVBench"

        logger.info(f"Loading LVBench meta from {meta_path}")

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Meta file not found at {meta_path}. Please check the path.")

        all_rows = []
        
        # 遍历数据列表，构建 DataFrame 所需的行
        for idx, item in enumerate(data_list):
            # 提取 video_info 中的嵌套信息，用于填充 metadata
            video_info = item.get("video_info", {})
            duration_minutes = video_info.get("duration_minutes")
            resolution = video_info.get("resolution", {})

            # 构建行数据
            row = {
                "index": idx,
                "key": item.get("key"),
                "type": item.get("type"),
                "uid": item.get("uid"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "question_type": item.get("question_type"),
                "time_reference": item.get("time_reference"),
                "video_path": item.get("video_path"),
                # 转换时长为秒（保留此字段以便后续统计或截断使用）
                "duration_seconds": duration_minutes * 60.0 if isinstance(duration_minutes, (int, float)) else None,
                "video_fps": video_info.get("fps"),
                "width": resolution.get("width"),
                "height": resolution.get("height"),
            }
            all_rows.append(row)

        # 转换为 DataFrame
        df = pd.DataFrame(all_rows)
        logger.info(f"Loaded LVBench dataset: {len(df)} samples (pre-processed)")

        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="LVBench",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            sample_path=sample_path,
            **kwargs,
        )

    def _build_struct(self, line):
        struct = []

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

        raw_q = line["question"].strip()
        question_text = (
                "Select the best answer to the following multiple-choice question based on the video. "
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
                f"Question: {raw_q}\n"
                "The best answer is:"
            )

        struct.append(dict(type="text", value=question_text))
        

        return struct

        # lines = raw_q.split("\n")

        # # 第一行是问题主体
        # main_question = lines[0].rstrip("?").rstrip()  # 可选：去掉末尾问号（非必须）

        # # 提取选项：从剩余行中找 A. B. C. D. 开头的
        # options = []
        # option_pattern = re.compile(r"^([A-D])\.\s*(.*)")
        # for l in lines[1:]:
        #     l = l.strip()
        #     if not l:
        #         continue
        #     match = option_pattern.match(l)
        #     if match:
        #         letter, text = match.groups()
        #         options.append(f"{letter}. {text}")
        #     else:
        #         pass

        # # 如果未能正确解析出4个选项，回退到原始方式（安全兜底）
        # if len(options) != 4:
        #     # 回退：直接使用原问题（保持兼容性）
        #     question_text = (
        #         "Select the best answer to the following multiple-choice question based on the video. "
        #         "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        #         f"Question: {raw_q}\n"
        #         "The best answer is:"
        #     )
        # else:
        #     # ✅ 严格按照目标格式构建
        #     question_text = (
        #         "Select the best answer to the following multiple-choice question based on the video. "
        #         "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        #         f"Question: {main_question}? Possible answer choices:\n"
        #         + "\n".join(options)
        #         + "\n"
        #         "The best answer is:"
        #     )

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
