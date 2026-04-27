import glob
import json
import logging
import os.path as osp

import numpy as np
import pandas as pd

# from vlmeval.smp import get_abs_path
from vlmeval.vlm.qwen3_vl.model import ensure_video_url

from .base import VideoDataset

logger = logging.getLogger(__name__)


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

class VideoMMMUDataset(VideoDataset):
    def __init__(self, processor, **kwargs):
        data_root = "/data/oceanus_share/shangshouduo-jk/myproject/data/processed/VideoMMMU"
        data_path = osp.join(data_root, "VideoMMMU.tsv")

        if not osp.exists(data_path):
            raise FileNotFoundError(f"VideoMMMU.tsv files found in {data_root}")
        logger.info(f"Loading VideoMMMU from {data_path}")
        df = pd.read_csv(data_path, sep="\t")

        # 确保 index 存在
        if "index" not in df.columns:
            df = df.reset_index(drop=True)
            df["index"] = np.arange(len(df))

        logger.info(f"Loaded VideoMMMU dataset: {len(df)} samples")

        sample_path = kwargs.pop("sample_path", None)
        super().__init__(
            processor,
            dataset_name="VideoMMMU",
            data_root=data_root,
            data_path=None,
            dataframe=df,
            sample_path=sample_path,
            **kwargs,
        )

    def _build_struct(self, line):
        struct = []
        # add video w/wo frame indices /images list
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

        question = line['question'] # 原始问题，包含 <image 1>
        category = line.get('category', '')
        q_type = line.get('question_type', 'open-ended') 
        # 从 TSV 读取的 line['options'] 是字符串，需要反序列化为 List
        raw_options = line.get('options', '[]')
        if isinstance(raw_options, str):
            try:
                # 优先尝试标准 JSON
                options_list = json.loads(raw_options)
            except:
                # 兼容旧脚本生成的单引号字符串
                try:
                    options_list = eval(raw_options)
                except:
                    options_list = []
                    logger.warning(f"Failed to parse options for index {line.get('index')}: {raw_options}")
        else:
            options_list = raw_options if isinstance(raw_options, list) else []

        image_video_path = line.get('image_path', None)

        # 定义 Prompt 常量
        PRE_PROMPT = "You should watch and learn the video content. Then apply what you learned to "
        PERCEPTION_PROMPT = "\nPlease ignore the Quiz question in last frame of the video."
        
        # Adaptation 专用 Prompt 后缀
        # 这句话，它建立了 <image 1> 和输入流中第二个视频的联系
        ADAPT_SUFFIX = "The image for this question is at the end of the video. " 
        final_text = ""

        # --- 逻辑分支 1: Adaptation (Interleaved / Image-Video) ---
        if 'Adaptation' in category:
            # 2. 输入 Image 视频 (模拟 <image> 位置)
            # 在 Qwen 的处理流中，这相当于 <video> <image> 的视觉输入顺序
            if image_video_path and pd.notna(image_video_path):
                struct.append({"type": "video", "value": image_video_path})
            else:
                logger.warning(f"Sample {line['index']} missing image_path")

            # 3. 构建文本
            # 格式: Pre-prompt + 指引话术 + 问题(含<image 1>) + 选项
            prompt_start = PRE_PROMPT
            
            # 拼接: "answer the following... question. The image... is at the end..."
            if q_type == 'multiple-choice':
                prompt_start += f"answer the following multi-choice question. {ADAPT_SUFFIX}"
                parsed_opts = parse_options_videommmu(options_list)
                question_body = f"{question}\n{parsed_opts}"
            else:
                prompt_start += f"answer the following open-ended question. {ADAPT_SUFFIX}"
                question_body = f"{question}"
            
            final_text = f"{prompt_start}{question_body}"

        # --- 逻辑分支 2: Perception & Comprehension ---
        else:
            if q_type == 'multiple-choice':
                parsed_opts = parse_options_videommmu(options_list)
                question_body = f"{question}\n{parsed_opts}"
            else:
                question_body = question
            
            final_text = f"{question_body}{PERCEPTION_PROMPT}"

        # 4. 强制 CoT 结尾
        final_text += '\nAdd `Answer: {Your final answer}` at the end of your reply.'
        # final_text += '\nThe best answer is:'
        
        struct.append({"type": "text", "value": final_text})
        
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger.info("VideoMMMU evaluation should be done offline using the VLMEvalKit specific script logic.")
        return None

    def evaluate_offline(self, file):
        return