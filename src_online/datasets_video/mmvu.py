import json
import logging
from pathlib import Path

import pandas as pd

from .base import VideoDataset

logger = logging.getLogger(__name__)


class MMVUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        self.use_cot = True

        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MMVU"
        json_path = Path(data_root) / "validation_process.json"

        with open(json_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        all_rows = []
        for idx, item in enumerate(data_list):
            all_rows.append(
                {
                    "index": idx,
                    "question_id": item.get("id", idx),
                    "video_path": item["video_path"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "choices": item.get("choices", None),
                    "answer": item["answer"],
                    "category": Path(item["video_path"]).parent.name,
                }
            )

        df = pd.DataFrame(all_rows)

        super().__init__(
            processor,
            dataset_name="MMVU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        q_type = line["question_type"]
        question = line["question"]

        if q_type == "multiple-choice":
            choices = line["choices"]
            if self.use_cot:
                prompt_text = (
                    f"Question: {question}\n"
                    f"A: {choices['A']}\n"
                    f"B: {choices['B']}\n"
                    f"C: {choices['C']}\n"
                    f"D: {choices['D']}\n"
                    f"E: {choices.get('E', '')}\n"
                    "Visual Information: processed video\n"
                    "Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. "
                    "Conclude by stating the final answer using the following format: \"Therefore, the final answer is: $LETTER\" (without quotes), "
                    "where $LETTER is one of the options. Think step by step before answering."
                )
            else:
                prompt_text = (
                    f"Question: {question}\n"
                    f"A: {choices['A']}\n"
                    f"B: {choices['B']}\n"
                    f"C: {choices['C']}\n"
                    f"D: {choices['D']}\n"
                    f"E: {choices.get('E', '')}\n"
                    "Visual Information: processed video\n"
                    "Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices."
                )
            prompt_text = prompt_text.replace("E: \n", "")
        else:
            if self.use_cot:
                prompt_text = (
                    f"\nQuestion: {question}\n"
                    "Visual Information: processed video\n"
                    "Answer the given question step by step. Begin by explaining your reasoning process clearly. "
                    "Conclude by stating the final answer using the following format: \"Therefore, the final answer is: Answer: $ANSWER\" "
                    "(without quotes), where $ANSWER is the final answer of the question. Think step by step before answering."
                )
            else:
                prompt_text = (
                    f"Question: {question}\n"
                    "Visual Information: processed video\n"
                    "Do not generate any intermediate reasoning process. Directly output the final answer."
                )

        return [
            {"type": "video", "value": line["video_path"]},
            {"type": "text", "value": prompt_text},
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger.info("MMVU evaluation requires external judge.")
        return None
