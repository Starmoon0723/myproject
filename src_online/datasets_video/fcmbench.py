import json
import logging

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class FCMBenchDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        meta_path = "/data/oceanus_share/cuirunze-jk/instructions/fcmbench_longvideo-en-v1.0_20260210.jsonl"
        data_root = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/FCMBench"

        data_list = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))

        all_rows = []
        for idx, item in enumerate(data_list):
            all_rows.append(
                {
                    "index": idx,
                    "question": item.get("prompt"),
                    "answer": item.get("answer"),
                    "video_path": item.get("video_path"),
                }
            )

        df = pd.DataFrame(all_rows)

        super().__init__(
            processor,
            dataset_name="FCMBench",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        video_path = "/data/oceanus_share/cuirunze-jk/video/视频数据集-en/" + line["video_path"]
        return [
            {"type": "video", "value": video_path},
            {"type": "text", "value": line["question"]},
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
