import json
import logging
import os.path as osp

import numpy as np
import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class CharadesSTADataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA"
        data_path = osp.join(data_root, "charades_sta_test.json")

        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        # 转成 DataFrame
        df = pd.DataFrame(data_list)
        if "index" not in df.columns:
            df["index"] = np.arange(len(df))

        super().__init__(
            processor,
            dataset_name="Charades-STA",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        video_path = osp.join(self.data_root, "videos", line["video"])
        question = line["question"]

        # 构建 prompt
        prompt_text = (
            "You are given a video and a question. "
            "Please provide the time segment in seconds when the action described in the question occurs.\n"
            f"Question: {question}\n"
            "Answer with the format: <start_time> - <end_time>"
        )

        struct = [
            {"type": "video", "value": video_path},
            {"type": "text", "value": prompt_text},
        ]
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
