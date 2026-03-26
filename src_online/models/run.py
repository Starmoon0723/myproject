import argparse
import logging
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_video import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def _parse_base_urls(base_urls: str):
    urls = [u.strip().rstrip("/") for u in base_urls.split(",") if u.strip()]
    if not urls:
        raise ValueError("No base URLs provided.")
    return urls


def worker(group_id, num_groups, args):
    evaluator = Evaluator(
        group_id=group_id,
        num_groups=num_groups,
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output,
        post_process=args.post_process,
        resume=not args.disable_resume,
        eval_only=False,
        base_url=args.base_url,
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        fps=args.fps,
    )
    evaluator.inference()


def main(args):
    base_urls = _parse_base_urls(args.base_urls)
    num_groups = args.nproc if args.nproc is not None else len(base_urls)

    logger.info("Launching %s workers over %s vLLM endpoints", num_groups, len(base_urls))

    processes = []
    for group_id in range(num_groups):
        base_url = base_urls[group_id % len(base_urls)]
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--group_id",
            str(group_id),
            "--num_groups",
            str(num_groups),
            "--base_url",
            base_url,
            "--model_name",
            args.model_name,
            "--model_path",
            args.model_path,
            "--dataset",
            args.dataset,
            "--output",
            args.output,
            "--api_key",
            args.api_key,
            "--request_timeout",
            str(args.request_timeout),
            "--max_retries",
            str(args.max_retries),
            "--fps",
            str(args.fps),
        ]

        if args.post_process:
            cmd.append("--post_process")
        if args.disable_resume:
            cmd.append("--disable_resume")

        p = subprocess.Popen(cmd, env=os.environ.copy())
        processes.append(p)
        logger.info("[Group %s] %s", group_id, base_url)

    for p in processes:
        p.wait()

    merge_evaluator = Evaluator(
        group_id=0,
        num_groups=num_groups,
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output,
        post_process=args.post_process,
        resume=not args.disable_resume,
        eval_only=True,
        base_url=base_urls[0],
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        fps=args.fps,
    )

    if merge_evaluator.merge_results():
        logger.info("Results merged successfully")
    else:
        logger.error("Failed to merge results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run online video evaluation with multiple vLLM endpoints")
    parser.add_argument("--nproc", type=int, default=None, help="Worker process count, default=len(base_urls)")
    parser.add_argument("--worker", action="store_true", help="Worker mode")
    parser.add_argument("--group_id", type=int, default=0)
    parser.add_argument("--num_groups", type=int, default=1)

    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="Video-MME")
    parser.add_argument("--output", type=str, default="./outputs/Qwen3-VL-online")

    parser.add_argument("--base_urls", type=str, default="http://127.0.0.1:22002/v1")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:22002/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--request_timeout", type=int, default=3600)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--fps", type=float, default=2)

    parser.add_argument("--post_process", action="store_true", default=False)
    parser.add_argument("--disable_resume", action="store_true", default=False)

    args = parser.parse_args()

    if args.worker:
        worker(args.group_id, args.num_groups, args)
    else:
        main(args)
