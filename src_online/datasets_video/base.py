import base64
import logging
import mimetypes
import os.path as osp
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return {
        "line": [item[0] for item in batch],
        "inputs": [item[1] for item in batch],
    }


def _video_path_to_data_uri(video_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"
    with open(video_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class VideoDataset(Dataset):
    def __init__(
        self,
        processor,
        group_id=0,
        num_groups=1,
        dataset_name=None,
        data_root=None,
        data_path=None,
        dataframe=None,
        fps=2,
        nframe=None,
        min_frames=4,
        max_frames=2048,
        system_prompt=None,
        completed_indices=None,
        sample_path=None,
    ):
        del processor

        self.group_id = group_id
        self.num_groups = num_groups
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.data_path = data_path
        self.sample_path = sample_path

        if dataframe is not None:
            full_data = dataframe
        else:
            logger.info("Loading data from %s", self.data_path)
            full_data = pd.read_csv(self.data_path, sep="\t")

        if completed_indices:
            original_count = len(full_data)
            full_data = full_data[~full_data["index"].isin(completed_indices)]
            logger.info(
                "[Group %s] Filtered %s completed samples, %s left",
                self.group_id,
                original_count - len(full_data),
                len(full_data),
            )

        if self.num_groups > 1:
            group_indices = list(range(group_id, len(full_data), num_groups))
            self.data = full_data.iloc[group_indices].reset_index(drop=True)
        else:
            self.data = full_data.reset_index(drop=True)

        self.system_prompt = system_prompt
        if fps is None and nframe is None:
            raise ValueError("At least one of fps or nframe must be set.")
        if fps is not None and nframe is not None:
            raise ValueError("Only one of fps or nframe can be set.")

        self.fps = fps
        self.nframe = nframe
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._video_data_uri_cache = {}

    def __len__(self):
        return len(self.data)

    def _build_struct(self, line):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        raise NotImplementedError

    def _build_messages(self, struct):
        content = []
        for item in struct:
            if item["type"] == "video":
                video_path = str(item["value"])
                abs_video_path = video_path
                if not osp.isabs(video_path) and self.data_root is not None:
                    abs_video_path = osp.join(self.data_root, video_path)
                abs_video_path = str(Path(abs_video_path))
                if abs_video_path not in self._video_data_uri_cache:
                    self._video_data_uri_cache[abs_video_path] = _video_path_to_data_uri(abs_video_path)
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": self._video_data_uri_cache[abs_video_path]},
                    }
                )
            elif item["type"] == "text":
                content.append({"type": "text", "text": str(item["value"])})
            else:
                raise ValueError(f"Unsupported content type: {item['type']}")

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    # def _build_mm_processor_kwargs(self):
    #     kwargs = {"do_sample_frames": True}
    #     if self.fps is not None:
    #         kwargs["fps"] = self.fps
    #     if self.nframe is not None:
    #         kwargs["num_frames"] = self.nframe
    #         kwargs["do_sample_frames"] = False
    #     if self.min_frames is not None:
    #         kwargs["min_frames"] = self.min_frames
    #     if self.max_frames is not None:
    #         kwargs["max_frames"] = self.max_frames
    #     return kwargs

    def __getitem__(self, index):
        line = self.data.iloc[index]
        struct = self._build_struct(line)
        request_payload = {
            "messages": self._build_messages(struct),
            # "extra_body": {"mm_processor_kwargs": self._build_mm_processor_kwargs()},
            "extra_body": {
                "media_io_kwargs": {
                    "video": {
                        "num_frames": 2048,
                        "fps": 2
                    }
                },
                "mm_processor_kwargs": {
                    "do_sample_frames": False,
                }
            }, 
        }
        return line, request_payload
