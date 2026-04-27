import json
import logging
import os
import os.path as osp
import sys
import copy
import hashlib
import re

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
        max_pixels=655360,  # 768、640，640*32*32,655360
        total_pixels=117440512,  # 224000*32*32
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
        mm_cache_dir=None,
        use_mm_cache=False,
        mm_cache_read_only=False,
    ):
        self.processor = processor
        self.group_id = group_id
        self.num_groups = num_groups
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.data_path = data_path
        self.sample_path = sample_path
        self.mm_cache_dir = mm_cache_dir
        self.use_mm_cache = bool(use_mm_cache and mm_cache_dir is not None)
        self.mm_cache_read_only = bool(mm_cache_read_only)
        if self.use_mm_cache:
            os.makedirs(self.mm_cache_dir, exist_ok=True)

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

            def _norm_video_key(p):
                if p is None:
                    return None
                try:
                    return str(Path(str(p))).replace("\\", "/").lower()
                except Exception:
                    return str(p).replace("\\", "/").lower()

            sample_by_video = {}
            for item in sample_data:
                if not isinstance(item, dict):
                    continue
                for key in ("source_video_rel_path", "video_path", "source_video_path"):
                    if key in item and item[key] is not None:
                        vk = _norm_video_key(item[key])
                        if vk and vk not in sample_by_video:
                            sample_by_video[vk] = item

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
                        if not item and "video_path" in r and sample_by_video:
                            vk = _norm_video_key(r["video_path"])
                            item = sample_by_video.get(vk, {})
                        out.append(item.get(field, default))
                    return out

                # support both legacy "frame_indices" and q-frame "selected_frame_indices"
                if any(("frame_indices" in it) or ("selected_frame_indices" in it) for it in sample_by_index.values()):
                    full_data["frame_indices"] = _get_field("frame_indices", None)
                    full_data["selected_frame_indices"] = _get_field("selected_frame_indices", None)

                if any(("video_token_keep_indices" in it) for it in sample_by_index.values()):
                    full_data["video_token_keep_indices"] = _get_field("video_token_keep_indices", None)
                if any(("video_token_pruning_stage" in it) for it in sample_by_index.values()):
                    full_data["video_token_pruning_stage"] = _get_field("video_token_pruning_stage", None)

                if any(("image_paths" in it) for it in sample_by_index.values()):
                    full_data["image_paths"] = _get_field("image_paths", None)

                logger.info(f"Successfully merged sample data (by index) from {sample_path} into full_data")
            elif len(sample_by_video) > 0:
                def _get_field_by_video(field: str, default):
                    out = []
                    for _, r in full_data.iterrows():
                        item = {}
                        if "video_path" in r:
                            vk = _norm_video_key(r["video_path"])
                            item = sample_by_video.get(vk, {})
                        out.append(item.get(field, default))
                    return out

                if any(("frame_indices" in it) or ("selected_frame_indices" in it) for it in sample_by_video.values()):
                    full_data["frame_indices"] = _get_field_by_video("frame_indices", None)
                    full_data["selected_frame_indices"] = _get_field_by_video("selected_frame_indices", None)

                if any(("video_token_keep_indices" in it) for it in sample_by_video.values()):
                    full_data["video_token_keep_indices"] = _get_field_by_video("video_token_keep_indices", None)
                if any(("video_token_pruning_stage" in it) for it in sample_by_video.values()):
                    full_data["video_token_pruning_stage"] = _get_field_by_video("video_token_pruning_stage", None)

                if any(("image_paths" in it) for it in sample_by_video.values()):
                    full_data["image_paths"] = _get_field_by_video("image_paths", None)

                logger.info(
                    f"Successfully merged sample data (by video_path) from {sample_path} into full_data"
                )
            else:
                if len(sample_data) == len(full_data):
                    full_data["frame_indices"] = [item.get("frame_indices", None) for item in sample_data]
                    full_data["selected_frame_indices"] = [item.get("selected_frame_indices", None) for item in sample_data]
                    full_data["video_token_keep_indices"] = [item.get("video_token_keep_indices", None) for item in sample_data]
                    full_data["video_token_pruning_stage"] = [item.get("video_token_pruning_stage", None) for item in sample_data]
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
        self.use_vllm = True
        self.frame_factor = 2

    @staticmethod
    def _safe_json_loads(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def _extract_video_token_keep_indices(self, line):
        return self._safe_json_loads(line.get("video_token_keep_indices", None))

    def _extract_video_token_pruning_stage(self, line):
        return self._safe_json_loads(line.get("video_token_pruning_stage", None))

    def _sample_index(self, line, fallback_index: int) -> int:
        try:
            return int(line.get("index", fallback_index))
        except Exception:
            return int(fallback_index)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = str(name).strip()
        if not name:
            return "unknown_video"
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
        safe = safe.strip("._")
        return safe or "unknown_video"

    @staticmethod
    def _stable_short_hash(obj) -> str:
        try:
            raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        except Exception:
            raw = str(obj)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]

    def _extract_frame_indices(self, line):
        # Use normalized values to avoid duplicate cache caused by string/list format differences.
        fi = self._safe_json_loads(line.get("frame_indices", None))
        if fi is None:
            return None
        if not isinstance(fi, (list, tuple)):
            return None
        out = []
        for x in fi:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out

    def _mm_cache_path(self, line, dataset_index: int) -> str:
        sample_idx = self._sample_index(line, dataset_index)
        dataset_tag = (self.dataset_name or "dataset").replace("/", "_").replace("\\", "_")
        video_path = line.get("video_path", None)
        if video_path is None:
            # fallback for non-video samples
            return osp.join(self.mm_cache_dir, dataset_tag, f"sample_{sample_idx}.pt")

        video_name = Path(str(video_path)).stem
        safe_video_name = self._sanitize_filename(video_name)
        video_path_hash = self._stable_short_hash(str(video_path).replace("\\", "/").lower())
        filename = f"{safe_video_name}__v_{video_path_hash}"

        # Keep cache robust when same video is evaluated with different frame selections.
        frame_indices = self._extract_frame_indices(line)
        if frame_indices is not None and len(frame_indices) > 0:
            filename = f"{filename}__fi_{self._stable_short_hash(frame_indices)}"

        return osp.join(self.mm_cache_dir, dataset_tag, f"{filename}.pt")

    def _prepare_vision_content(self, content):
        # Cache should be strategy-agnostic: remove token-pruning selection fields.
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                item2 = dict(item)
                item2.pop("video_token_keep_indices", None)
                out.append(item2)
            else:
                out.append(item)
        return out

    def _build_messages(self, struct, *, for_vision: bool):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        content = self._prepare_content(struct, dataset_name=self.dataset_name)
        if for_vision:
            content = self._prepare_vision_content(content)
        messages.append({"role": "user", "content": content})
        return messages

    def _compute_vision_tensors(self, messages):
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_kwargs is None:
            video_kwargs = {}
        video_kwargs["do_resize"] = False
        return image_inputs, video_inputs, video_kwargs

    def _load_mm_cache(self, cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        image_inputs = payload.get("image_inputs", None)
        video_inputs = payload.get("video_inputs", None)
        video_kwargs = payload.get("video_kwargs", {}) or {}
        return image_inputs, video_inputs, video_kwargs

    def _save_mm_cache(self, cache_path, image_inputs, video_inputs, video_kwargs):
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        payload = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "video_kwargs": video_kwargs,
        }
        torch.save(payload, cache_path)

    def _inject_video_token_keep_indices(self, video_inputs, token_keep_indices):
        if video_inputs is None:
            return None
        if token_keep_indices is None:
            return video_inputs
        # One sample only supports one video in current pipeline.
        injected = []
        for idx, (video_tensor, metadata) in enumerate(video_inputs):
            md = dict(metadata) if isinstance(metadata, dict) else copy.deepcopy(metadata)
            if idx == 0:
                md["video_token_keep_indices"] = token_keep_indices
            injected.append((video_tensor, md))
        return injected

    def _inject_video_token_pruning_stage(self, video_inputs, token_pruning_stage):
        if video_inputs is None:
            return None
        if token_pruning_stage is None:
            return video_inputs
        injected = []
        for idx, (video_tensor, metadata) in enumerate(video_inputs):
            md = dict(metadata) if isinstance(metadata, dict) else copy.deepcopy(metadata)
            if idx == 0:
                md["video_token_pruning_stage"] = token_pruning_stage
            injected.append((video_tensor, md))
        return injected

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
            raise ValueError("Do not mix 'video' and 'image' inputs in the same sample.")
        if len(video_inputs) == 0 and len(image_inputs) == 0:
            raise ValueError("No visual input found. Expect at least one 'video' or one/more 'image'.")
        if len(video_inputs) > 1:
            raise ValueError("Only support 1 video input.")
        for s in inputs:
            if s["type"] == "video":
                item = {"type": "video", "video": ensure_video_url(s["value"])}
                # 添加帧索引信息
                if "frame_indices" in s:
                    item["frame_indices"] = s["frame_indices"]
                if "video_token_keep_indices" in s:
                    item["video_token_keep_indices"] = s["video_token_keep_indices"]
                if "video_token_pruning_stage" in s:
                    item["video_token_pruning_stage"] = s["video_token_pruning_stage"]
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

        messages = self._build_messages(struct, for_vision=False)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs = None
        video_inputs = None
        video_kwargs = {}
        cache_path = self._mm_cache_path(line, index) if self.use_mm_cache else None
        if cache_path is not None and osp.exists(cache_path):
            image_inputs, video_inputs, video_kwargs = self._load_mm_cache(cache_path)
        else:
            vision_messages = self._build_messages(struct, for_vision=True)
            image_inputs, video_inputs, video_kwargs = self._compute_vision_tensors(vision_messages)
            if cache_path is not None and not self.mm_cache_read_only:
                self._save_mm_cache(cache_path, image_inputs, video_inputs, video_kwargs)

        token_keep_indices = self._extract_video_token_keep_indices(line)
        video_inputs = self._inject_video_token_keep_indices(video_inputs, token_keep_indices)
        token_pruning_stage = self._extract_video_token_pruning_stage(line)
        video_inputs = self._inject_video_token_pruning_stage(video_inputs, token_pruning_stage)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        return line, inputs
