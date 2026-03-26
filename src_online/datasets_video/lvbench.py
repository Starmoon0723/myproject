import json
import logging

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class LVBenchDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/LVBench/video_info.meta.json"
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/LVBench"

        with open(meta_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        all_rows = []
        for idx, item in enumerate(data_list):
            all_rows.append(
                {
                    "index": idx,
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "video_path": item.get("video_path"),
                }
            )

        df = pd.DataFrame(all_rows)

        super().__init__(
            processor,
            dataset_name="LVBench",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        question_text = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"Question: {line['question']}\n"
            "The best answer is:"
        )
        return [
            {"type": "video", "value": line["video_path"]},
            {"type": "text", "value": question_text},
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
