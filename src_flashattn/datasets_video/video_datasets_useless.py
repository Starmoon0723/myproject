import glob
import json
import logging
import os
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vlmeval.smp import *
from vlmeval.vlm.qwen3_vl.model import ensure_video_url
from vlmeval.dataset.videomme import VideoMME
from vlmeval.dataset.mvbench import MVBench_MP4

# 1. 定义自定义文件所在的文件夹路径
custom_module_path = "/data/oceanus_share/shangshouduo-jk/project/code/frame_extraction"

# 2. 将该路径加入到系统搜索路径中
if custom_module_path not in sys.path:
    sys.path.append(custom_module_path)

# 3. 直接导入文件名作为模块
from overwrite_vision_process import process_vision_info

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return {
        "line": [item[0] for item in batch],
        "inputs": [item[1] for item in batch],
    }


def parse_options_videommmu(options):
    """VideoMMMU 专用选项格式化函数"""
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            return options  # Fallback

    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    # 检查是否已经包含 A. B. 等前缀
    if all(opt.startswith(f"{letter}.") for opt, letter in zip(options, option_letters)):
        return "\n".join(options)

    choices_str = "\n".join([f"{letter}. {opt}" for letter, opt in zip(option_letters, options)])
    return choices_str


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

            # 确保长度一致，按序添加 frame_indices
            if len(sample_data) == len(full_data):
                frame_indices_list = [item.get("frame_indices", []) for item in sample_data]
                full_data["frame_indices"] = frame_indices_list
                logger.info(f"Successfully added frame_indices from {sample_path} to full_data")
            else:
                logger.warning(
                    f"Sample data length ({len(sample_data)}) does not match full_data length ({len(full_data)}). "
                    "Skipping frame_indices assignment."
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
        video_count = len(video_inputs)
        assert video_count == 1, "Only support 1 video input."
        for s in inputs:
            if s["type"] == "video":
                item = {"type": "video", "video": ensure_video_url(s["value"])}
                # 添加帧索引信息
                if "frame_indices" in s:
                    item["frame_indices"] = s["frame_indices"]
                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None:
                    item["total_pixels"] = self.total_pixels
                if self.min_frames is not None:
                    item["min_frames"] = self.min_frames
                if self.max_frames is not None:
                    item["max_frames"] = self.max_frames
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
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
                content.append(item)
            elif s["type"] == "audio":
                item = {"type": "audio", "audio": s["value"]}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        # Print the final content for debugging
        logger.info(
            f"Prepared content for dataset {dataset_name}:\n{json.dumps(content, ensure_ascii=False, indent=2)}"
        )
        return content

    def __getitem__(self, index):
        line = self.data.iloc[index]
        struct = self._build_struct(line)

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(struct, dataset_name=self.dataset_name)})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        logger.info(f"DEBUG videos: {videos[0][0].shape}")
        logger.info(f"[DBG] videos type: {type(videos)}")
        if videos is not None:
            logger.info(f"[DBG] len(videos): {len(videos)}")
            v0 = videos[0]
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
            if images is not None:
                mm_data["image"] = images
            if videos is not None:
                # 注意：这里用 process_vision_info 返回的 videos 原样传入（包含 metadata）
                mm_data["video"] = videos

            mm_processor_kwargs = dict(video_kwargs)
            mm_processor_kwargs["do_resize"] = False  # 如需强制关闭 resize

            inputs = {
                "prompt": text,                 # 关键：用 prompt 字符串，而不是 prompt_token_ids
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": mm_processor_kwargs,
            }
            if videos is not None:
                videos_only, video_metadatas = zip(*videos)
                videos_only = list(videos_only)
                video_metadatas = list(video_metadatas)
                logger.info(f"[DBG] video_metadatas: {video_metadatas}")
           
        else:
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None
            inputs = self.processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
        
        return line, inputs


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
        sample_path = "/data/oceanus_share/shangshouduo-jk/project/code/frame_extraction/dataset/method/aks/videomme/selected_frames_offical/videomme/blip/selected_frames.jsonl"
        super().__init__(processor, dataset_name="Video-MME", data_root=data_root, data_path=data_path, sample_path=sample_path, **kwargs)

    def _build_struct(self, line):
        struct = []
        fi = line.get("frame_indices", None)

        # struct.append(dict(type='video', value=osp.join(self.data_root, line['video_path'])))
        v = dict(type="video", value=osp.join(self.data_root, line["video_path"]))
        if fi is not None:
            v["frame_indices"] = fi

        struct.append(v)
        question = "\n" + line["question"] + "\n" + "\n".join(eval(line["candidates"]))
        struct.append(dict(type="text", value=f"\nThese are the frames of a video. {question}\nThe best answer is: "))
        return struct

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        judge_kwargs = dict(model="qwen__qwen3-vl-8b-instruct", nproc=1)
        return VideoMME.evaluate(eval_file, **judge_kwargs)


class MLVUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        # MLVU 数据集根目录
        data_root = "/data/oceanus_share/dukang-jk/datasets/MVLU/MLVU"

        # 1. 读取所有子任务的 JSON 文件并合并
        json_dir = osp.join(data_root, "json_new_answer")
        # json_dir = osp.join(data_root, 'json_max_10_video')
        json_files = sorted(glob.glob(osp.join(json_dir, "*.json")))

        all_data = []
        global_idx = 0

        for j_file in json_files:
            # 从文件名获取任务名称 (例如 "1_plotQA.json" -> "1_plotQA")
            # 这对应了 video/ 下的子文件夹名称
            task_name = osp.splitext(osp.basename(j_file))[0]

            with open(j_file, "r", encoding="utf-8") as f:
                data_list = json.load(f)

            for item in data_list:
                # 添加必须的元数据
                item["task_folder"] = task_name
                # 为每条数据生成全局唯一的 index (VideoDataset 需要根据 index 过滤已完成数据)
                if "index" not in item:
                    item["index"] = global_idx
                all_data.append(item)
                global_idx += 1

        # 转换为 DataFrame
        df = pd.DataFrame(all_data)
        logger.info(f"Loaded MLVU dataset: {len(df)} samples from {len(json_files)} tasks.")

        # kwargs['max_frames'] = 2048
        # kwargs['fps'] = 2

        # 调用父类，传入构建好的 dataframe
        super().__init__(
            processor,
            dataset_name="MLVU",
            data_root=data_root,
            data_path=None,  # 不需要路径，因为传了 dataframe
            dataframe=df,  # 传入内存中的数据
            **kwargs,
        )

    def _build_struct(self, line):
        # 1. 构建视频绝对路径: root/video/task_folder/video_filename
        video_path = osp.join(self.data_root, "video", line["task_folder"], line["video"])

        # 2. 处理选项
        # 数据格式为字符串: "['A. Yellow', 'B. Red']" -> 需要 eval 转回 list
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
        # struct.append(dict(type='video', value=video_path, duration=line.get('duration', None)))    # 添加duration参数
        struct.append(dict(type="video", value=video_path))
        struct.append(dict(type="text", value=question_text))

        # Print the struct for debugging purposes
        # logger.info(f"Built structure for MLVUDataset:\n{json.dumps(struct, ensure_ascii=False, indent=2)}")
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None


class LVBenchDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/LVBench"
        meta_path = osp.join(data_root, "video_info.meta.new.jsonl")
        video_dir = osp.join(data_root, "videos")

        # 缺失视频 key 的记录文件（放在数据集目录下）
        missing_key_path = osp.join(data_root, "missing_video_keys.txt")

        all_rows = []
        global_idx = 0

        missing_keys = []
        total_video_lines = 0
        skipped_video_lines = 0
        total_qa = 0
        kept_qa = 0

        def resolve_video_path(video_key: str):
            """
            返回 video_path 或 None（缺失则 None）
            不再 raise，避免整个评测中断
            """
            # common case: {key}.mp4
            p1 = osp.join(video_dir, f"{video_key}.mp4")
            if osp.exists(p1):
                return p1

            # fallback: any mp4 contains the key
            hits = sorted(glob.glob(osp.join(video_dir, f"*{video_key}*.mp4")))
            if hits:
                return hits[0]

            # last resort: any file contains the key
            hits2 = sorted(glob.glob(osp.join(video_dir, f"*{video_key}*.*")))
            if hits2:
                return hits2[0]

            return None

        logger.info(f"Loading LVBench meta from {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total_video_lines += 1
                obj = json.loads(line)

                key = obj.get("key")
                vtype = obj.get("type", None)
                video_info = obj.get("video_info", {}) or {}
                duration_minutes = video_info.get("duration_minutes", None)
                fps = video_info.get("fps", None)
                resolution = video_info.get("resolution", {}) or {}
                width = resolution.get("width", None)
                height = resolution.get("height", None)

                video_path = resolve_video_path(key)
                qa_list = obj.get("qa", [])

                # 统计 QA 总量（即使缺视频也统计一下，便于你知道损失多少）
                if isinstance(qa_list, list):
                    total_qa += len(qa_list)

                # 缺视频：整条跳过
                if video_path is None:
                    skipped_video_lines += 1
                    missing_keys.append(key)
                    logger.warning(f"[LVBench] Missing video for key={key}, skip this video and all its QA")
                    continue

                if not isinstance(qa_list, list):
                    continue

                # flatten qa
                for qa in qa_list:
                    if not isinstance(qa, dict):
                        continue

                    row = {
                        "index": global_idx,
                        "key": key,
                        "type": vtype,
                        "uid": qa.get("uid", None),
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "question_type": qa.get("question_type", None),
                        "time_reference": qa.get("time_reference", None),
                        "video_path": video_path,
                        "duration_seconds": duration_minutes * 60.0 if isinstance(duration_minutes, (int, float)) else None,
                        "video_fps": fps,
                        "width": width,
                        "height": height,
                    }
                    all_rows.append(row)
                    global_idx += 1
                    kept_qa += 1

        # 写缺失 key 清单（去重 + 保序）
        if missing_keys:
            seen = set()
            uniq_missing = []
            for k in missing_keys:
                if k not in seen:
                    seen.add(k)
                    uniq_missing.append(k)

            with open(missing_key_path, "w", encoding="utf-8") as wf:
                for k in uniq_missing:
                    wf.write(str(k) + "\n")

            logger.warning(f"[LVBench] Missing videos: {len(uniq_missing)} keys. Saved to {missing_key_path}")

        logger.info(
            f"[LVBench] meta video lines={total_video_lines}, skipped video lines={skipped_video_lines}, "
            f"total QA={total_qa}, kept QA={kept_qa}"
        )

        df = pd.DataFrame(all_rows)
        logger.info(f"Loaded LVBench dataset: {len(df)} samples (flattened, missing videos skipped)")

        super().__init__(
            processor,
            dataset_name="LVBench",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        video_path = line["video_path"]

        raw_q = line["question"].strip()
        lines = raw_q.split("\n")

        # 第一行是问题主体
        main_question = lines[0].rstrip("?").rstrip()  # 可选：去掉末尾问号（非必须）

        # 提取选项：从剩余行中找 A. B. C. D. 开头的
        options = []
        option_pattern = re.compile(r"^([A-D])\.\s*(.*)")
        for l in lines[1:]:
            l = l.strip()
            if not l:
                continue
            match = option_pattern.match(l)
            if match:
                letter, text = match.groups()
                options.append(f"{letter}. {text}")
            else:
                # 如果某行不是标准选项，但看起来像选项（比如没有字母），可选择跳过或警告
                # 这里保守处理：一旦格式不对，回退到原始方式（避免出错）
                # 但 LVBench 应该是规范的，所以可忽略
                pass

        # 如果未能正确解析出4个选项，回退到原始方式（安全兜底）
        if len(options) != 4:
            # 回退：直接使用原问题（保持兼容性）
            question_text = (
                "Select the best answer to the following multiple-choice question based on the video. "
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
                f"Question: {raw_q}\n"
                "The best answer is:"
            )
        else:
            # ✅ 严格按照目标格式构建
            question_text = (
                "Select the best answer to the following multiple-choice question based on the video. "
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
                f"Question: {main_question}? Possible answer choices:\n"
                + "\n".join(options)
                + "\n"
                "The best answer is:"
            )

        struct = [
            {"type": "video", "value": video_path},
            {"type": "text", "value": question_text},
        ]
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None


class MVBenchDataset(VideoDataset):
    """
    Local MVBench loader:
    - Read multiple json files under /data/.../MVBench/json_new_change/*.json
    - Each json is a list of samples:
        {
          "video": "...webm/mp4/avi",
          "question": "...",
          "candidates": ["A. ...", "B. ...", ...],
          "answer": "C",
          "video_path": "/abs/path/to/video"
        }
    - Prompt aligned with MLVU / LVBench.
    - Skip missing video files and optionally write missing list.
    """

    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench"
        json_dir = osp.join(data_root, "json_new_change")
        json_files = sorted(glob.glob(osp.join(json_dir, "*.json")))

        if not json_files:
            raise FileNotFoundError(f"[MVBench] No json files found under: {json_dir}")

        missing_video_path = osp.join(data_root, "missing_video_paths.txt")

        all_rows = []
        global_idx = 0

        missing_paths = []
        total_samples = 0
        kept_samples = 0

        logger.info(f"[MVBench] Loading jsons from: {json_dir}, files={len(json_files)}")

        for jf in json_files:
            task_name = osp.splitext(osp.basename(jf))[0]

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
            except Exception as e:
                logger.warning(f"[MVBench] Failed to load {jf}: {e}")
                continue

            if not isinstance(data_list, list):
                logger.warning(f"[MVBench] Invalid json content (not list): {jf}")
                continue

            for item in data_list:
                total_samples += 1
                if not isinstance(item, dict):
                    continue

                video_path = item.get("video_path", None)
                if not video_path:
                    # 兼容：如果有人没写 video_path，则尝试用 data_root 拼
                    vname = item.get("video", None)
                    if vname:
                        video_path = osp.join(data_root, "video", vname)

                # 缺失视频：跳过，避免中断
                if not video_path or not osp.exists(video_path):
                    missing_paths.append(str(video_path))
                    continue

                row = {
                    "index": global_idx,
                    "task": task_name,
                    "video": item.get("video", osp.basename(video_path)),
                    "video_path": video_path,
                    "question": item.get("question", ""),
                    "candidates": item.get("candidates", []),
                    "answer": item.get("answer", ""),
                }
                all_rows.append(row)
                global_idx += 1
                kept_samples += 1

        if missing_paths:
            with open(missing_video_path, "w", encoding="utf-8") as f:
                for p in missing_paths:
                    f.write(str(p) + "\n")
            logger.warning(
                f"[MVBench] Missing videos: {len(missing_paths)} samples (of {total_samples}). "
                f"Saved to {missing_video_path}"
            )

        df = pd.DataFrame(all_rows)
        logger.info(
            f"[MVBench] Loaded {len(df)} samples, kept {kept_samples}/{total_samples} "
            f"({kept_samples / max(total_samples, 1):.2%})"
        )

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

        candidates_list = line["candidates"]
        if isinstance(candidates_list, str):
            try:
                candidates_list = eval(candidates_list)
            except Exception:
                candidates_list = [candidates_list]

        options_str = "\n".join(candidates_list)

        question_text = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"Question: {line['question']} Possible answer choices:\n"
            f"{options_str}\n"
            "The best answer is:"
        )

        struct = [
            {"type": "video", "value": video_path},
            {"type": "text", "value": question_text},
        ]
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        judge_kwargs = dict(model="qwen__qwen3-vl-8b-instruct", nproc=1)
        return MVBench_MP4.evaluate(eval_file, **judge_kwargs)


class CharadesSTADataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA"
        data_path = osp.join(data_root, "charades_sta_test.json")

        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        # 转成 DataFrame
        df = pd.DataFrame(data_list)
        if "index" not in df.columns:
            df["index"] = np.arange(len(df))

        super().__init__(
            processor,
            dataset_name="Charades-STA",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        video_path = osp.join(self.data_root, "videos", line["video"])
        question = line["question"]

        # 构建 prompt
        prompt_text = (
            "You are given a video and a question. "
            "Please provide the time segment in seconds when the action described in the question occurs.\n"
            f"Question: {question}\n"
            "Answer with the format: <start_time> - <end_time>"
        )

        struct = [
            {"type": "video", "value": video_path},
            {"type": "text", "value": prompt_text},
        ]
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return None


class MMVUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MMVU"
        data_path = osp.join(data_root, "mmvu_test.json")

        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        # 转成 DataFrame
        df = pd.DataFrame(data_list)
        if "index" not in df.columns:
            df["index"] = np.arange(len(df))

        super().__init__(
            processor,
            dataset_name="MMVU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            **kwargs,
        )

    def _build_struct(self, line):
        # 视频路径
        video_path = osp.join(self.data_root, "videos", line["video"])

        # question
        question = line["question"]
        q_type = line.get("question_type", "multiple-choice")
        choices = line.get("choices", {})

        # === 构建 prompt ===
        if q_type == "multiple-choice":
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
            # 移除空的选项 (如果E不存在)
            prompt_text = prompt_text.replace("E: \n", "")

        else:
            # Open-ended
            prompt_text = (
                f"Question: {question}\n"
                "Visual Information: processed video\n"
                "Do not generate any intermediate reasoning process. Directly output the final answer."
            )

        struct = [
            {"type": "video", "value": video_path},
            {"type": "text", "value": prompt_text},
        ]
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        # 这里你可以实现简单的规则评测，或者留空
        logger.info("MMVU evaluation requires Hybrid Judge (Rule + GPT). Please run external eval script.")
        return None


class VideoMMMUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/VideoMMMU"
        data_path = osp.join(data_root, "VideoMMMU.tsv")

        # 如果 TSV 不存在，尝试直接读取 parquet (兼容逻辑)
        if not osp.exists(data_path):
            parquet_files = sorted(glob.glob(osp.join(data_root, "**/*.parquet"), recursive=True))
            if parquet_files:
                logger.info(f"TSV not found, loading from {len(parquet_files)} parquet files...")
                dfs = [pd.read_parquet(f) for f in parquet_files]
                df = pd.concat(dfs, ignore_index=True)
                # 简单处理 category
                if "category" not in df.columns:
                    df["category"] = "Unknown"
            else:
                raise FileNotFoundError(f"Neither VideoMMMU.tsv nor parquet files found in {data_root}")
        else:
            logger.info(f"Loading VideoMMMU from {data_path}")
            df = pd.read_csv(data_path, sep="\t")

        # 确保 index 存在
        if "index" not in df.columns:
            df = df.reset_index(drop=True)
            df["index"] = np.arange(len(df))

        # 处理 Adaptation 的 Image Video 路径
        if "image_path" in df.columns:
            df["image_path_abs"] = df["image_path"].apply(get_abs_path)
        else:
            df["image_path_abs"] = None

        logger.info(f"Loaded VideoMMMU dataset: {len(df)} samples")

        super().__init__(
            processor,
            dataset_name="VideoMMMU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            # VideoMMMU 视频长度差异大，建议 fps=1 或 2，或者由基类自适应
            # fps=1,
            **kwargs,
        )

    def _prepare_content(self, inputs, dataset_name=None):
        content = []
        for s in inputs:
            if s["type"] == "video":
                item = {"type": "video", "video": ensure_video_url(s["value"])}
                # 复用基类的像素和帧数配置
                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None:
                    item["total_pixels"] = self.total_pixels
                if self.min_frames is not None:
                    item["min_frames"] = self.min_frames
                if self.max_frames is not None:
                    item["max_frames"] = self.max_frames

                if self.fps is not None:
                    item["fps"] = self.fps
                elif self.nframe is not None:
                    # 如果是按帧数采样，这里保留基类逻辑
                    import cv2

                    video = cv2.VideoCapture(s["value"])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.frame_factor * self.frame_factor
                        item["nframes"] = new_frame_count
                    else:
                        item["nframes"] = self.nframe

                content.append(item)

            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")

        return content

    def _build_struct(self, line):
        video_path = line["video_path"]
        question = line["question"]
        category = line.get("category", "")
        q_type = line.get("question_type", "open-ended")  # multiple-choice 或 open-ended
        options = line.get("options", [])

        # --- 复刻 VLMEvalKit 的 Prompt 逻辑 ---

        # 定义 Prompt 模板
        PRE_PROMPT = "You should watch and learn the video content. Then apply what you learned to "
        PERCEPTION_AND_COMPREHENSION_PROMPT = "\nPlease ignore the Quiz question in last frame of the video."
        MCQ_PROMPT = "answer the following multi-choice question. The image for this question is at the end of the video.\n"
        OPEN_ENDED_PROMPT = "answer the following open-ended question. The image for this question is at the end of the video.\n"

        final_text = ""

        # 逻辑分支 1: Adaptation 类任务 (通常涉及图文交错，这里简化为 Video + Text)
        if "Adaptation" in category:
            prompt_start = PRE_PROMPT
            if q_type == "multiple-choice":
                prompt_start += MCQ_PROMPT
                parsed_opts = parse_options_videommmu(options)
                question_body = f"{question}\n{parsed_opts}"
            else:
                prompt_start += OPEN_ENDED_PROMPT
                question_body = question

            final_text = f"{prompt_start}{question_body}"

        # 逻辑分支 2: Perception & Comprehension 类任务
        else:
            post_prompt = PERCEPTION_AND_COMPREHENSION_PROMPT
            if q_type == "multiple-choice":
                parsed_opts = parse_options_videommmu(options)
                question_body = f"{question}\n{parsed_opts}"
            else:
                question_body = question

            final_text = f"{question_body}{post_prompt}"

        # 强制 CoT 引导，方便后续提取
        final_text += "\nAdd `Answer: {Your final answer}` at the end of your reply."

        struct = [
            {"type": "video", "value": video_path},
            {"type": "text", "value": final_text},
        ]
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger.info("VideoMMMU evaluation should be done offline using the VLMEvalKit specific script logic.")
        return None
