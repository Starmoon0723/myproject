import os

os.environ["NCCL_DEBUG"] = ""
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import subprocess
import sys

import numpy as np
import torch
import torch.multiprocessing

import logging

import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # 覆盖可能存在的默认/第三方配置，确保日志可见
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_video import Evaluator

# random.seed(3407)  # 设置随机种子
# np.random.seed(3407)
# torch.manual_seed(3407)
# torch.cuda.manual_seed_all(3407)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def worker(group_id, num_groups, args):
    random.seed(3407)  # 设置随机种子
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

    logger.info(f"[Group {group_id}] Process initialized, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    evaluator = Evaluator(
        group_id=group_id,
        num_groups=num_groups,
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output,
        moe=args.moe,
        post_process=args.post_process,
        resume=not args.disable_resume,
        eval_only=False,  # Worker需要完整初始化
        gpu_memory_utilization=args.gpu_memory_utilization,
        sample_path=args.sample_path,
    )

    # Worker只执行推理
    evaluator.inference()


def main(nproc):
    num_groups = nproc
    num_gpus = torch.cuda.device_count()
    world_size = num_gpus // num_groups

    logger.info(f"Launching {num_groups} groups...")

    processes = []
    for group_id in range(num_groups):
        gpu_start = group_id * world_size
        gpu_end = (group_id + 1) * world_size
        gpu_ids = ",".join(map(str, range(gpu_start, gpu_end)))

        # 1. 获取当前可见的物理设备列表（例如外部传入了 "2,3"）
        current_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        if current_visible_devices:
            # 如果外部有指定，我们需要做映射
            device_list = current_visible_devices.split(',')
            # 获取当前组需要的逻辑索引（比如 0,1）
            logical_indices = range(gpu_start, gpu_end)
            # 将逻辑索引映射回物理索引（比如 0->2, 1->3）
            mapped_ids = [device_list[i] for i in logical_indices]
            gpu_ids = ",".join(mapped_ids)
        else:
            # 如果外部没指定，就直接用数字
            gpu_ids = ",".join(map(str, range(gpu_start, gpu_end)))
        
        # 创建环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env["WORLD_SIZE"] = str(world_size)
        logger.info(f"[Group {group_id}] WORLD_SIZE={world_size}, CUDA_VISIBLE_DEVICES={gpu_ids}")

        # 使用subprocess启动新的Python进程
        cmd = [
            sys.executable,  # Python解释器路径
            __file__,  # 当前脚本
            "--worker",  # 标识这是worker进程
            "--group_id",
            str(group_id),
            "--num_groups",
            str(num_groups),
            "--model_name",
            args.model_name,
            "--model_path",
            args.model_path,
            "--dataset",
            args.dataset,
            "--output",
            args.output,
        ]
        if args.sample_path:
            cmd.extend(["--sample_path", args.sample_path])
        cmd.extend(["--gpu_memory_utilization", str(args.gpu_memory_utilization)])

        if args.moe:
            cmd.append("--moe")
        if args.post_process:
            cmd.append("--post_process")
        if args.disable_resume:
            cmd.append("--disable_resume")

        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    merge_evaluator = Evaluator(
        group_id=0,
        num_groups=num_groups,
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output,
        moe=args.moe,
        post_process=args.post_process,
        resume=not args.disable_resume,
        eval_only=True,  # 使用eval_only模式，不初始化模型和数据
        sample_path=args.sample_path,
    )

    # 等待所有进程完成
    for p in processes:
        p.wait()

    logger.info("All worker processes completed!")

    # 所有worker完成后，由主进程执行合并和评估（使用eval_only模式）
    logger.info("Starting merge and evaluation...")
    # 执行合并
    if merge_evaluator.merge_results():
        logger.info("Results merged successfully!")
        # 执行评估
        # merge_evaluator.evaluate()
    else:
        logger.info("Failed to merge results!")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Run video evaluation with Qwen3-VL model")
    parser.add_argument("--nproc", type=int, default=1)                         # 进程数
    parser.add_argument("--group_id", type=int, default=0)                      # 组id
    parser.add_argument("--num_groups", type=int, default=1)                    # 组数
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct") # 模型名称
    parser.add_argument("--model_path", type=str, default="/data/oceanus_ctr/j-cuirunze-jk/ckpts/Qwen/Qwen3-VL-8B-Instruct") # 模型路径
    parser.add_argument("--dataset", type=str, default="Video-MME")              # 数据集名称
    parser.add_argument("--output", type=str, default="/data/oceanus_ctr/j-shangshouduo-jk/myproject/output/results/backbone")      # 输出路径
    parser.add_argument("--moe", action="store_true", default=False)             # 是否使用moe
    parser.add_argument("--post_process", action="store_true", default=False)    # 处理cot输出
    parser.add_argument("--disable_resume", action="store_true", default=False)  # 是否禁用续写
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)     # gpu内存利用率
    parser.add_argument(
        "--sample_path",
        type=str,
        default=None,
        help="Optional jsonl path to per-sample frame_indices/image_paths (e.g. q-frame selected_frames_manifest.jsonl).",
    )

    args = parser.parse_args()

    main(args.nproc)

