import json
import logging
import os.path as osp
import re

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class MVBenchDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/hfmodel/datasets/MVBench"
        data_path = "/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/hfmodel/datasets/processed/MVBench/mvbench_eval_clean_tvqa_mp4_exist.jsonl"

        rows = []
        logger.info(f"[MVBench] Loading data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item.setdefault("index", idx)
                rows.append(item)

        dataframe = pd.DataFrame(rows)
        logger.info(f"[MVBench] Loaded {len(dataframe)} samples")

        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="MVBench",
            data_root=data_root,
            data_path=data_path,
            dataframe=dataframe,
            sample_path=sample_path,
            **kwargs,
        )

    @staticmethod
    def _json_loads_if_needed(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def _resolve_video_path(self, video_path):
        video_path = str(video_path)
        if video_path.startswith(("file://", "http://", "https://")) or osp.isabs(video_path):
            return video_path
        return osp.join(self.data_root, video_path)

    def _build_struct(self, line):
        struct = []
        image_paths = self._json_loads_if_needed(line.get("image_paths", None))

        if isinstance(image_paths, list) and len(image_paths) > 0:
            for p in image_paths:
                struct.append(dict(type="image", value=str(p)))
        else:
            fi = self._json_loads_if_needed(line.get("frame_indices", None))
            tk = self._json_loads_if_needed(line.get("video_token_keep_indices", None))
            ts = self._json_loads_if_needed(line.get("video_token_pruning_stage", None))

            video_path = self._resolve_video_path(line["video_path"])
            video_item = dict(type="video", value=video_path)
            if fi is not None:
                video_item["frame_indices"] = fi
            if tk is not None:
                video_item["video_token_keep_indices"] = tk
            if ts is not None:
                video_item["video_token_pruning_stage"] = ts

            start = line.get("start", None)
            end = line.get("end", None)
            if start is None or end is None:
                bound = self._json_loads_if_needed(line.get("bound", None))
                if isinstance(bound, (list, tuple)) and len(bound) == 2:
                    start, end = bound
            if start is not None:
                video_item["video_start"] = float(start)
            if end is not None:
                video_item["video_end"] = float(end)
            struct.append(video_item)

        prompt = line.get("prompt", None)
        if not isinstance(prompt, str) or not prompt.strip():
            option_lines = self._json_loads_if_needed(line.get("option_lines", None))
            if not isinstance(option_lines, list) or len(option_lines) == 0:
                options = self._json_loads_if_needed(line.get("options", None))
                if isinstance(options, dict):
                    option_lines = [f"{k}. {v}" for k, v in sorted(options.items())]
                else:
                    candidates = self._json_loads_if_needed(line.get("candidates", []))
                    option_lines = [
                        f"{chr(ord('A') + i)}. {candidate}"
                        for i, candidate in enumerate(candidates)
                    ]
            prompt = (
                f"Question: {line['question']}\n"
                "Options:\n"
                f"{chr(10).join(str(x) for x in option_lines)}\n"
                "Only give the best option."
            )

        text = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the option letter (A, B, C, or D).\n"
            f"{prompt}\n"
            "The best answer is:"
        )
        struct.append(dict(type="text", value=text))
        return struct

    @staticmethod
    def _extract_option(value):
        if value is None:
            return None
        text = str(value).strip().upper()
        match = re.search(r"\b([A-D])\b", text)
        if match:
            return match.group(1)
        match = re.search(r"^\(?([A-D])[\).:：、\s]", text)
        if match:
            return match.group(1)
        return text[:1] if text[:1] in {"A", "B", "C", "D"} else None

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        if eval_file.endswith((".xlsx", ".xls")):
            df = pd.read_excel(eval_file)
        else:
            rows = []
            with open(eval_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)

        if "answer_option" not in df.columns or "prediction" not in df.columns:
            raise ValueError("MVBench evaluation requires `answer_option` and `prediction` columns.")

        pred = df["prediction"].map(cls._extract_option)
        gt = df["answer_option"].astype(str).str.strip().str.upper()
        correct = pred == gt
        summary = {
            "total": int(len(df)),
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()) if len(df) > 0 else 0.0,
        }
        if "task_type" in df.columns:
            summary["per_task"] = {}
            for task, task_df in df.assign(_correct=correct).groupby("task_type"):
                summary["per_task"][str(task)] = {
                    "total": int(len(task_df)),
                    "correct": int(task_df["_correct"].sum()),
                    "accuracy": float(task_df["_correct"].mean()) if len(task_df) > 0 else 0.0,
                }
        return summary
