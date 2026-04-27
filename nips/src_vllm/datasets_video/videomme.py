import json
import logging
import os
import os.path as osp
from pathlib import Path

from vlmeval.dataset.videomme import VideoMME

from .base import VideoDataset

logger = logging.getLogger(__name__)


class VideoMMEDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        # 需要修改
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/Video-MME"
        # 该位置设置文件名称
        # data_path = osp.join(data_root, 'Video-MME.tsv')
        # data_path = osp.join(data_root, 'Video-MME_max_video.tsv')
        data_path = osp.join(data_root, "Video-MME_copy.tsv")
        # data_path = "/data/oceanus_share/shangfangxin-jk/projects/Video-MME-Sample.tsv"
        # sample_path = "/data/oceanus_share/shangshouduo-jk/project/code/frame_extraction/dataset/Video-MME-sorted_output.jsonl"
        # sample_path = "/data/oceanus_share/shangshouduo-jk/project/code/frame_extraction/dataset/method/aks/videomme/selected_frames_offical/videomme/blip/selected_frames.jsonl"
        # sample_path = kwargs.pop("sample_path", None)
        # sample_path = "/data/oceanus_share/shangshouduo-jk/project/code/frame_extraction/dataset/method/aks/videomme/selected_frames_offical/videomme/clip/videomme_clip_64.jsonl"
        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="Video-MME",
            data_root=data_root,
            data_path=data_path,
            sample_path=sample_path,
            **kwargs,
        )

    def _build_struct(self, line):
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
            tk = line.get("video_token_keep_indices", None)
            ts = line.get("video_token_pruning_stage", None)
            if isinstance(tk, str):
                try:
                    tk = json.loads(tk)
                except Exception:
                    tk = None
            if isinstance(ts, str):
                try:
                    ts = json.loads(ts)
                except Exception:
                    pass
            v = dict(type="video", value=osp.join(self.data_root, line["video_path"]))
            if fi is not None:
                v["frame_indices"] = fi
            if tk is not None:
                v["video_token_keep_indices"] = tk
            if ts is not None:
                v["video_token_pruning_stage"] = ts
            struct.append(v)

        question = "\n" + line["question"] + "\n" + "\n".join(eval(line["candidates"]))
        struct.append(dict(type="text", value=f"\nThese are the frames of a video. {question}\nThe best answer is: "))
        return struct

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        # judge_kwargs = dict(model="qwen__qwen3-vl-8b-instruct", nproc=1)
        # return VideoMME.evaluate(eval_file, **judge_kwargs)
        return
