import ast
import logging
import os.path as osp

# from vlmeval.dataset.videomme import VideoMME

from .base import VideoDataset

logger = logging.getLogger(__name__)


class VideoMMEDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_ctr/j-shangshouduo-jk/myproject/data/processed/Video-MME"
        data_path = osp.join(data_root, "Video-MME.tsv")
        super().__init__(
            processor,
            dataset_name="Video-MME",
            data_root=data_root,
            data_path=data_path,
            **kwargs,
        )

    def _build_struct(self, line):
        video_path = osp.join("/data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/data", line["video_path"])
        candidates = line["candidates"]
        if isinstance(candidates, str):
            candidates = ast.literal_eval(candidates)
        question = "\n" + line["question"] + "\n" + "\n".join(candidates)
        return [
            {"type": "video", "value": video_path},
            {"type": "text", "value": f"\nThese are the frames of a video. {question}\nThe best answer is: "},
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        # judge_kwargs = dict(model="qwen__qwen3-vl-8b-instruct", nproc=1)
        # return VideoMME.evaluate(eval_file, **judge_kwargs)
        return None
