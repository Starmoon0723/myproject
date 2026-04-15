import glob
import json
import logging
import os.path as osp

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class FCMBenchDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        # meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/FCMBench/fcmbench_longvideo_v1.0_20260205_absolute.jsonl"
        # meta_path = "/data/oceanus_share/cuirunze-jk/instructions/fcmbench_longvideo_v1.0_20260210.jsonl"
        meta_path = "/data/oceanus_share/cuirunze-jk/instructions/fcmbench_longvideo-en-v1.0_20260210.jsonl"
        data_root = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/FCMBench"

        logger.info(f"Loading FCMBench meta from {meta_path}")

        data_list = []
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    # print(f"读取行: {line}")  # 打印每一行
                    line = line.strip()
                    data_list.append(json.loads(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"Meta file not found at {meta_path}. Please check the path.")

        all_rows = []
        
        for idx, item in enumerate(data_list):
            row = {
                "index": idx,
                "task_id": item.get("task_id"),
                "difficulty": item.get("difficulty"), # 原始文件名，留作参考
                "question": item.get("prompt"),
                "answer": item.get("answer"),
                "task_category": item.get("task_category"),
                "video_path": item.get("video_path"),
            }
            all_rows.append(row)

        # 转换为 DataFrame
        df = pd.DataFrame(all_rows)
        logger.info(f"Loaded FCMBench dataset: {len(df)} samples (pre-processed)")

        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="FCMBench",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            sample_path=sample_path,
            **kwargs,
        )

    def _build_struct(self, line):


        question_text = line["question"]

        # 3. 组装结构
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
            # v = dict(type="video", value="/data/oceanus_share/cuirunze-jk/video/视频拼接3/" + line["video_path"])
            v = dict(type="video", value="/data/oceanus_share/cuirunze-jk/video/视频拼接-en/" + line["video_path"])
            if fi is not None:
                v["frame_indices"] = fi
            struct.append(v)
        struct.append(dict(type="text", value=question_text))
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
