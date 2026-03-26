import ast
import json
import logging
import os.path as osp

import numpy as np
import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


def parse_options_videommmu(options):
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            return options

    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    if all(opt.startswith(f"{letter}.") for opt, letter in zip(options, option_letters)):
        return "\n".join(options)

    return "\n".join([f"{letter}. {opt}" for letter, opt in zip(option_letters, options)])


class VideoMMMUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/VideoMMMU"
        data_path = osp.join(data_root, "VideoMMMU.tsv")

        if not osp.exists(data_path):
            raise FileNotFoundError(f"VideoMMMU.tsv not found in {data_root}")

        df = pd.read_csv(data_path, sep="\t")
        if "index" not in df.columns:
            df = df.reset_index(drop=True)
            df["index"] = np.arange(len(df))

        super().__init__(
            processor,
            dataset_name="VideoMMMU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        struct = [{"type": "video", "value": line["video_path"]}]

        question = line["question"]
        category = line.get("category", "")
        q_type = line.get("question_type", "open-ended")
        raw_options = line.get("options", "[]")

        if isinstance(raw_options, str):
            try:
                options_list = json.loads(raw_options)
            except Exception:
                try:
                    options_list = ast.literal_eval(raw_options)
                except Exception:
                    options_list = []
        else:
            options_list = raw_options if isinstance(raw_options, list) else []

        pre_prompt = "You should watch and learn the video content. Then apply what you learned to "
        perception_prompt = "\nPlease ignore the Quiz question in last frame of the video."
        adapt_suffix = "The image for this question is at the end of the video. "

        if "Adaptation" in category:
            if q_type == "multiple-choice":
                parsed_opts = parse_options_videommmu(options_list)
                question_body = f"{question}\n{parsed_opts}"
                final_text = f"{pre_prompt}answer the following multi-choice question. {adapt_suffix}{question_body}"
            else:
                final_text = f"{pre_prompt}answer the following open-ended question. {adapt_suffix}{question}"
        else:
            if q_type == "multiple-choice":
                parsed_opts = parse_options_videommmu(options_list)
                question_body = f"{question}\n{parsed_opts}"
            else:
                question_body = question
            final_text = f"{question_body}{perception_prompt}"

        final_text += "\nAdd `Answer: {Your final answer}` at the end of your reply."
        struct.append({"type": "text", "value": final_text})
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger.info("VideoMMMU evaluation should be done offline via dedicated script.")
        return None
