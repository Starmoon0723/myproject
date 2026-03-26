import ast
import glob
import json
import logging
import os.path as osp

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class MVBenchDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MVBench/mvbench_all.json"
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench"

        with open(meta_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        all_rows = []
        for idx, item in enumerate(data_list):
            all_rows.append(
                {
                    "index": idx,
                    "video": item.get("video"),
                    "video_path": item.get("video_path"),
                    "question": item.get("question"),
                    "candidates": item.get("candidates"),
                    "answer": item.get("answer"),
                }
            )

        df = pd.DataFrame(all_rows)

        super().__init__(
            processor,
            dataset_name="MVBench",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        video_path = line["video_path"]
        if video_path == "none":
            tvqa_task_path = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench/video/unpacked/tvqa/frames_fps3_hq/"
            frame_dir = tvqa_task_path + line["video"]
            mp4_path = f"{frame_dir}.mp4"
            if not osp.exists(mp4_path):
                from moviepy.editor import ImageSequenceClip

                frames = sorted(glob.glob(osp.join(frame_dir, "*.jpg")))
                clip = ImageSequenceClip(frames, fps=3)
                clip.write_videofile(mp4_path, codec="libx264", logger=None)
            video_path = mp4_path

        candidates_list = line["candidates"]
        if isinstance(candidates_list, str):
            candidates_list = ast.literal_eval(candidates_list)

        options_str = "\n".join(candidates_list)
        question_text = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"Question: {line['question']} Possible answer choices:\n"
            f"{options_str}\n"
            "The best answer is:"
        )

        return [
            {"type": "video", "value": video_path},
            {"type": "text", "value": question_text},
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
