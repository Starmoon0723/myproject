import json
import logging
import os
import os.path as osp
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset

from vlmeval.vlm.qwen3_vl.model import ensure_video_url
from pathlib import Path

custom_module_path = "/data/oceanus_share/shangshouduo-jk/myproject/src/models/ours/code"
if custom_module_path not in sys.path:
    sys.path.append(custom_module_path)
from overwrite_vision_process import process_vision_info

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return {
        "line": [item[0] for item in batch],
        "inputs": [item[1] for item in batch],
    }


class VideoDataset(Dataset):
    def __init__(
        self,
        processor,
        group_id=0,
        num_groups=1,
        dataset_name=None,
        data_root=None,
        data_path=None,
        dataframe=None,  # 添加dataframe参数，优先使用传入的 DataFrame 而不是读取文件
        min_pixels=49152,  # 128*32*32，48*32*32
        max_pixels=786432,  # 768、640，640*32*32,655360
        total_pixels=234881024,  # 224000*32*32
        image_min_pixels=3136,
        image_max_pixels=12845056,
        fps=None,
        nframe=None,
        min_frames=4,
        max_frames=2048,
        system_prompt=None,
        use_vllm=False,
        completed_indices=None,
        sample_path=None,
    ):
        self.processor = processor
        self.group_id = group_id
        self.num_groups = num_groups
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.data_path = data_path
        self.sample_path = sample_path

        if dataframe is not None:
            logger.info("Loading data from DataFrame provided in memory")
            full_data = dataframe
        else:
            logger.info(f"Loading data from {self.data_path}")
            full_data = pd.read_csv(self.data_path, sep="\t")
        if sample_path is not None:
            logger.info(f"Loading sample data from {sample_path}")
            with open(sample_path, "r", encoding="utf-8") as f:
                sample_data = [json.loads(line) for line in f]

            # Prefer index-based alignment if possible; fallback to positional (must match length).
            sample_by_index = None
            if len(sample_data) > 0 and isinstance(sample_data[0], dict) and "index" in sample_data[0]:
                try:
                    sample_by_index = {int(item["index"]): item for item in sample_data if "index" in item}
                except Exception:
                    sample_by_index = None

            if sample_by_index is not None:
                def _get_field(field: str, default):
                    out = []
                    for _, r in full_data.iterrows():
                        idx = int(r["index"])
                        item = sample_by_index.get(idx, {})
                        out.append(item.get(field, default))
                    return out

                # support both legacy "frame_indices" and q-frame "selected_frame_indices"
                if any(("frame_indices" in it) or ("selected_frame_indices" in it) for it in sample_by_index.values()):
                    full_data["frame_indices"] = _get_field("frame_indices", None)
                    full_data["selected_frame_indices"] = _get_field("selected_frame_indices", None)

                if any(("image_paths" in it) for it in sample_by_index.values()):
                    full_data["image_paths"] = _get_field("image_paths", None)

                logger.info(f"Successfully merged sample data (by index) from {sample_path} into full_data")
            else:
                if len(sample_data) == len(full_data):
                    full_data["frame_indices"] = [item.get("frame_indices", None) for item in sample_data]
                    full_data["selected_frame_indices"] = [item.get("selected_frame_indices", None) for item in sample_data]
                    full_data["image_paths"] = [item.get("image_paths", None) for item in sample_data]
                    logger.info(f"Successfully merged sample data (by order) from {sample_path} into full_data")
                else:
                    logger.warning(
                        f"Sample data length ({len(sample_data)}) does not match full_data length ({len(full_data)}). "
                        "Skipping sample merge."
                    )
        # 先过滤掉已完成的数据，得到剩余待处理的数据
        if completed_indices is not None and len(completed_indices) > 0:
            original_count = len(full_data)
            full_data = full_data[~full_data["index"].isin(completed_indices)]
            filtered_count = len(full_data)
            logger.info(
                f"[Group {self.group_id}] Filtered out {original_count - filtered_count} completed samples, "
                f"{filtered_count} remaining in total"
            )

        # 根据group_id和num_groups对剩余数据进行均匀分配 - 轮询分配
        if self.num_groups > 1:
            group_indices = list(range(group_id, len(full_data), num_groups))
            self.data = full_data.iloc[group_indices].reset_index(drop=True)
            logger.info(
                f"[Group {self.group_id}] processing {len(self.data)} samples with round-robin assignment "
                "from remaining data"
            )
        else:
            self.data = full_data.reset_index(drop=True)

        self.system_prompt = system_prompt
        assert fps is None or nframe is None, "fps和nframe只能设置一项"
        assert fps is not None or nframe is not None, "fps和nframe必须设置一项"

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.fps = fps
        self.nframe = nframe
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.use_vllm = use_vllm
        self.frame_factor = 2

    def __len__(self):
        return len(self.data)

    def _build_struct(self, line):
        raise NotImplementedError

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError

    def _prepare_content(self, inputs, dataset_name=None):
        content = []
        video_inputs = [s for s in inputs if s["type"] == "video"]
        image_inputs = [s for s in inputs if s["type"] == "image"]

        # For q-frame reproduction we support multi-image OR single-video.
        if len(video_inputs) > 0 and len(image_inputs) > 0:
            if self.dataset_name != "VideoMMMU" :
                raise ValueError("Do not mix 'video' and 'image' inputs in the same sample.")
        if len(video_inputs) == 0 and len(image_inputs) == 0:
            raise ValueError("No visual input found. Expect at least one 'video' or one/more 'image'.")
        if len(video_inputs) > 1:
            if self.dataset_name != "VideoMMMU" :
                raise ValueError("Only support 1 video input.")
        for s in inputs:
            if s["type"] == "video":
                item = {"type": "video", "video": ensure_video_url(s["value"])}
                # 添加帧索引信息
                if "frame_indices" in s:
                    item["frame_indices"] = s["frame_indices"]
                if self.min_pixels is not None: item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None: item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None: item["total_pixels"] = self.total_pixels
                if self.min_frames is not None: item["min_frames"] = self.min_frames
                if self.max_frames is not None: item["max_frames"] = self.max_frames
                
                if self.fps is not None:
                    item["fps"] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s["value"])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.frame_factor * self.frame_factor
                        logger.info(f"use {new_frame_count} for {s['value']}")
                        item["nframes"] = new_frame_count
                    else:
                        item["nframes"] = self.nframe
                content.append(item)
            elif s["type"] == "image":
                # q-frame: multi-image input; allow per-image pixel constraints to preserve multi-res benefit
                item = {"type": "image", "image": s["value"]}
                if self.image_min_pixels is not None: item["min_pixels"] = self.image_min_pixels
                if self.image_max_pixels is not None: item["max_pixels"] = self.image_max_pixels
                # optional overrides
                for k in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
                    if k in s:
                        item[k] = s[k]
                content.append(item)
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
                content.append(item)
            elif s["type"] == "audio":
                item = {"type": "audio", "audio": s["value"]}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        # Print the final content for debugging
        # logger.info(f"Prepared content for dataset {dataset_name}:\n{json.dumps(content, ensure_ascii=False, indent=2)}")
        return content

    def __getitem__(self, index):
        line = self.data.iloc[index]
        struct = self._build_struct(line)

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(struct, dataset_name=self.dataset_name)})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        logger.info(f"DEBUG text: {text}")
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        logger.info(f"[DBG] videos type: {type(video_inputs)}")
        if video_inputs is not None:
            logger.info(f"[DBG] len(videos): {len(video_inputs)}")
            v0 = video_inputs[0]
            logger.info(f"[DBG] videos[0] type: {type(v0)}")
            # 常见情况：v0 可能是 (frames_tensor, metadata) 或者其他结构
            try:
                if isinstance(v0, (tuple, list)) and hasattr(v0[0], "shape"):
                    logger.info(f"[DBG] frames shape: {v0[0].shape}")  # 重点看 T 维
                elif hasattr(v0, "shape"):
                    logger.info(f"[DBG] frames shape: {v0.shape}")
            except Exception as e:
                logger.info(f"[DBG] shape inspect failed: {e}")
        logger.info(f"[DBG] video_kwargs: {video_kwargs}")

        if self.use_vllm:
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            video_kwargs['do_resize'] = False # 不再做resize，因为process_vision_info已经做了resize

            inputs = {
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }
            if video_inputs is not None:
                _, video_metadatas = zip(*video_inputs)
                video_metadatas = list(video_metadatas)
                logger.info(f"[DBG] video_metadatas: {video_metadatas}")
        else:
            if video_inputs is not None:
                videos, video_metadatas = zip(*video_inputs)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )

        return line, inputs
