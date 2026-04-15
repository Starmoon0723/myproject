import glob
import json
import logging
import os.path as osp

import pandas as pd

from vlmeval.dataset.mvbench import MVBench_MP4

from .base import VideoDataset

logger = logging.getLogger(__name__)


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
        meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MVBench/mvbench_all.json"
        # meta_path = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MVBench/json_new_change/episodic_reasoning.json"
        
        data_root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench"

        logger.info(f"[MVBench] Loading meta from {meta_path}")

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Meta file not found at {meta_path}. Please check the path.")

        all_rows = []
        
        # 遍历数据列表
        for idx, item in enumerate(data_list):
            row = {
                "index": idx,
                # 如果预处理文件中没有 task 字段，给一个默认值，或者你可以从 json 文件名预处理进去
                # "task": item.get("task", "unknown"), 
                "video": item.get("video"),
                "video_path": item.get("video_path"),
                "question": item.get("question"),
                "candidates": item.get("candidates"), # 这是一个 list，例如 ["A. ...", "B. ..."]
                "answer": item.get("answer"),
            }
            all_rows.append(row)

        df = pd.DataFrame(all_rows)
        logger.info(f"[MVBench] Loaded {len(df)} samples (pre-processed)")

        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="MVBench",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            sample_path=sample_path,
            **kwargs,
        )

    def _build_struct(self, line):
        # tvqa task "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench/video/unpacked/tvqa/frames_fps3_hq"
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
            video_path = line["video_path"]
            if video_path == "none":
                tvqa_task_path = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench/video/unpacked/tvqa/frames_fps3_hq/"
                frame_dir = tvqa_task_path + line["video"]
                mp4_path = f"{frame_dir}.mp4"
                if not osp.exists(mp4_path):
                    # 用 moviepy 合成
                    # pip install moviepy==1.0.3
                    from moviepy.editor import ImageSequenceClip
                    print(frame_dir)
                    frames = sorted(glob.glob(osp.join(frame_dir, "*.jpg")))
                    clip = ImageSequenceClip(frames, fps=3)
                    clip.write_videofile(mp4_path, codec="libx264", logger=None)
                video_path = mp4_path
            else:
                video_path = video_path

            fi = line.get("frame_indices", None)
            v = dict(type="video", value=video_path)
            if fi is not None:
                v["frame_indices"] = fi
            struct.append(v)

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

        struct.append({"type": "text", "value": question_text})
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        return
