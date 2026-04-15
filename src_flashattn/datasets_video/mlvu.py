import glob
import json
import logging
import os.path as osp

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class MLVUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MLVU/mlvu_all.json"
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVLU/MLVU"

        logger.info(f"Loading MLVU meta from {meta_path}")

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Meta file not found at {meta_path}. Please check the path.")

        all_rows = []
        
        for idx, item in enumerate(data_list):
            row = {
                "index": idx,
                "video": item.get("video"), # 原始文件名，留作参考
                "duration": item.get("duration"),
                "question": item.get("question"),
                "candidates": item.get("candidates"), # 注意：这里仍然是字符串格式 "['A...', 'B...']"
                "answer": item.get("answer"),
                "question_type": item.get("question_type"),
                "video_path": item.get("video_path"),
            }
            all_rows.append(row)

        # 转换为 DataFrame
        df = pd.DataFrame(all_rows)
        logger.info(f"Loaded MLVU dataset: {len(df)} samples (pre-processed)")

        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="MLVU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            sample_path=sample_path,
            **kwargs,
        )

    def _build_struct(self, line):
        # 数据格式为字符串: "['A. Yellow', 'B. Red']" -> 需要 eval 转回 list
        # try:
        #     candidates_list = ast.literal_eval(line["candidates"])
        # except (ValueError, SyntaxError):
        #     # 如果解析失败，回退到 eval 或者直接处理
        candidates_list = eval(line["candidates"])

        options_str = "\n".join(candidates_list)

        question_text = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"Question: {line['question']} Possible answer choices:\n"
            f"{options_str}\n"
            "The best answer is:"
        )

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
            v = dict(type="video", value=line["video_path"])
            if fi is not None:
                v["frame_indices"] = fi
            struct.append(v)
        struct.append(dict(type="text", value=question_text))
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None
