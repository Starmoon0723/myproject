import ast
import json
import logging

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class MLVUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MLVU/mlvu_all.json"
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVLU/MLVU"

        with open(meta_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        all_rows = []
        for idx, item in enumerate(data_list):
            all_rows.append(
                {
                    "index": idx,
                    "video_path": item.get("video_path"),
                    "question": item.get("question"),
                    "candidates": item.get("candidates"),
                    "answer": item.get("answer"),
                }
            )

        df = pd.DataFrame(all_rows)

        super().__init__(
            processor,
            dataset_name="MLVU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        candidates = line["candidates"]
        if isinstance(candidates, str):
            candidates = ast.literal_eval(candidates)
        options_str = "\n".join(candidates)
        question_text = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"Question: {line['question']} Possible answer choices:\n"
            f"{options_str}\n"
            "The best answer is:"
        )
        return [
            {"type": "video", "value": line["video_path"]},
            {"type": "text", "value": question_text},
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
