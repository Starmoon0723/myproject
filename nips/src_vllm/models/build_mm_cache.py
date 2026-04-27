import os
import argparse
import logging

from modelscope import AutoProcessor
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets_video import VideoMMEDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build offline mm_data/video_kwargs cache for Video-MME.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--sample_path", type=str, default=None)
    parser.add_argument("--mm_cache_dir", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining samples.")
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.mm_cache_dir, exist_ok=True)

    logger.info(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)

    dataset = VideoMMEDataset(
        processor,
        group_id=0,
        num_groups=1,
        use_vllm=True,
        fps=args.fps,
        sample_path=args.sample_path,
        mm_cache_dir=args.mm_cache_dir,
        use_mm_cache=True,
        mm_cache_read_only=False,
    )

    total = len(dataset)
    start = max(0, int(args.start))
    end = total if args.limit <= 0 else min(total, start + int(args.limit))
    if start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, total={total}")

    logger.info(f"Building mm cache for range [{start}, {end}) / {total}")
    for i in tqdm(range(start, end), desc="Building mm cache"):
        _line, _inputs = dataset[i]
        # Accessing dataset[i] with use_mm_cache=True automatically computes and saves cache if missing.

    logger.info("mm cache build finished.")


if __name__ == "__main__":
    main()

