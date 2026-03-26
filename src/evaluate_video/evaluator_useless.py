import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["NCCL_DEBUG"] = ""
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import os.path as osp
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from modelscope import AutoProcessor, Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration
from torch.utils.data import DataLoader

from vlmeval.smp import *
from tqdm import tqdm

from datasets_video import (
    collate_fn,
    VideoMMEDataset,
    MVBenchDataset,
    MLVUDataset,
    LVBenchDataset,
    CharadesSTADataset,
    MMVUDataset,
    VideoMMMUDataset,
)

import logging

import random

random.seed(3407)  # 设置随机种子
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class Evaluator:
    def __init__(
        self,
        group_id=0,
        num_groups=1,
        model_name="Qwen3-VL-8B-Instruct",
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        dataset_name="Video-MME",
        output_dir="outputs",
        moe=False,
        use_vllm=False,
        post_process=False,
        resume=False,
        eval_only=False,  # 新增：只用于评估，不初始化模型和数据
        gpu_memory_utilization=0.5,
    ):
        self.group_id = group_id
        self.num_groups = num_groups
        self.model_name = f"{model_name}" if use_vllm else model_name
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.use_vllm = use_vllm
        self.resume = resume
        self.eval_only = eval_only
        self.gpu_memory_utilization = gpu_memory_utilization

        if self.group_id == 0:
            logger.info(f"Model name: {self.model_name}")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Dataset name: {self.dataset_name}")

        date, commit_id = timestr("day"), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        # self.pred_root = osp.join(output_dir, self.model_name, eval_id)
        self.pred_root = osp.join(output_dir, self.model_name, self.dataset_name, eval_id)
        if self.num_groups > 1:
            self.result_file_path = osp.join(
                self.pred_root, f"{self.group_id}{self.num_groups}_{self.model_name}_{self.dataset_name}.jsonl"
            )
            self.final_result_file_path = osp.join(self.pred_root, f"{self.model_name}_{self.dataset_name}.xlsx")
        else:
            self.result_file_path = osp.join(self.pred_root, f"{self.model_name}_{self.dataset_name}.jsonl")
            self.final_result_file_path = osp.join(self.pred_root, f"{self.model_name}_{self.dataset_name}.xlsx")
        os.makedirs(self.pred_root, exist_ok=True)

        # 设置数据集类
        if dataset_name == "Video-MME":
            self.VIDEO_DATASET_CLS = VideoMMEDataset
        elif dataset_name == "MVBench":
            self.VIDEO_DATASET_CLS = MVBenchDataset
        elif dataset_name == "MLVU":
            self.VIDEO_DATASET_CLS = MLVUDataset
        elif dataset_name == "LVBench":
            self.VIDEO_DATASET_CLS = LVBenchDataset
        elif dataset_name == "Charades-STA":
            self.VIDEO_DATASET_CLS = CharadesSTADataset
        elif dataset_name == "MMVU":
            self.VIDEO_DATASET_CLS = MMVUDataset
        elif dataset_name == "VideoMMMU":
            self.VIDEO_DATASET_CLS = VideoMMMUDataset
        else:
            raise ValueError("只支持Video-MME、MVBench、MLVU、LVBench、Charades-STA、MMVU、VideoMMMU")

        # 如果是eval_only模式，跳过所有耗时的初始化
        if eval_only:
            logger.info(f"[Group {self.group_id}] Running in eval_only mode, skipping model and data initialization")
            self.processor = None
            self.dataloader = None
            self.dataloader_iter = None
            self.model = None
            self.llm = None
            return

        # 以下是正常模式的初始化
        self.processor = AutoProcessor.from_pretrained(model_path)

        # 获取已完成的数据索引
        completed_indices = self._get_completed_indices()

        dataset = self.VIDEO_DATASET_CLS(
            self.processor,
            group_id=group_id,
            num_groups=num_groups,
            use_vllm=use_vllm,
            fps=2,
            # nframe=64,
            completed_indices=completed_indices,
        )
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=int(os.environ["WORLD_SIZE"]),
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context="spawn",  # fork
        )
        # 创建迭代器预热数据
        self.dataloader_iter = iter(self.dataloader)

        logger.info(f"Initializing model from {self.model_path}")
        if use_vllm:
            logger.info("Im using vllm")
            from vllm import LLM, SamplingParams

            video_limit = 2 if self.dataset_name == 'VideoMMMU' else 1
            limit_mm_per_prompt = {
                'image': 64,
                'video': video_limit
            }
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=1,
                max_model_len=262144,
                limit_mm_per_prompt=limit_mm_per_prompt,
                tensor_parallel_size=int(os.environ["WORLD_SIZE"]),
                gpu_memory_utilization=self.gpu_memory_utilization,
                seed=3407,
                mm_encoder_tp_mode="weights",
                enable_expert_parallel=moe,
                enable_chunked_prefill=True,
                max_num_batched_tokens=262144,
                skip_mm_profiling=True,
            )
            self.sampling_params = SamplingParams(
                top_p=0.8,  # 1.0
                top_k=20,  # 40
                temperature=0.7,  # 1.0
                repetition_penalty=1.0,
                presence_penalty=1.5,  # 2.0
                max_tokens=10,  # 32768
                stop_token_ids=None,
                logprobs=20,
            )

            if self.dataset_name == "Charades-STA":
                self.sampling_params.max_tokens = 32
            elif self.dataset_name == "MMVU":
                self.sampling_params.max_tokens = 32768
            elif self.dataset_name == "VideoMMMU":
                self.sampling_params.max_tokens = 32768

            # 因为 Qwen 的 tokenizer 可能会把 "A" 和 " A" (带空格) 分成不同 id
            if self.use_vllm and self.dataset_name not in ["Charades-STA", "MMVU", "VideoMMMU"]:
                # 获取 vllm 内部的 tokenizer
                tokenizer = self.llm.get_tokenizer()
                self.option_tokens = {}
                for char in ["A", "B", "C", "D"]:
                    # 寻找 "A" 和 " A" 对应的 id
                    ids = []
                    # 尝试编码 "A"
                    ids.append(tokenizer.encode(char, add_special_tokens=False)[0])
                    # 尝试编码 " A"
                    ids.append(tokenizer.encode(" " + char, add_special_tokens=False)[0])
                    # 去重并保存
                    self.option_tokens[char] = list(set(ids))
                logger.info(f"Target Option IDs: {self.option_tokens}")
        else:
            logger.info("Im using flash-attn")
            if moe:
                MODEL_CLS = Qwen3VLMoeForConditionalGeneration
            else:
                MODEL_CLS = Qwen3VLForConditionalGeneration
            try:
                self.model = MODEL_CLS.from_pretrained(
                    self.model_path,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
            except ImportError as e:
                logger.warning(f"flash_attention_2 unavailable ({e}), fallback to sdpa")
                self.model = MODEL_CLS.from_pretrained(
                    self.model_path,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="sdpa",  # 或者 "eager"
                )
            self.generation_config = dict(
                do_sample=True,
                top_p=0.8,
                top_k=20,
                temperature=0.7,
                repetition_penalty=1.0,
                max_new_tokens=10,  # 40960
            )
            if self.dataset_name == "Charades-STA":
                self.generation_config["max_new_tokens"] = 32
            elif self.dataset_name == "MMVU":
                self.generation_config["max_new_tokens"] = 32768
            elif self.dataset_name == "VideoMMMU":
                self.generation_config["max_new_tokens"] = 32768

        self.post_process = post_process

    def _get_completed_indices(self):
        """检查输出文件并返回已完成的数据索引（读取所有group的结果文件）"""
        completed_indices = set()

        if not self.resume:
            return completed_indices

        # 检查所有group的输出文件
        if self.num_groups > 1:
            for group_id in range(self.num_groups):
                group_result_file = osp.join(
                    self.pred_root, f"{group_id}{self.num_groups}_{self.model_name}_{self.dataset_name}.jsonl"
                )
                if osp.exists(group_result_file):
                    try:
                        group_completed = 0
                        with open(group_result_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    data = json.loads(line.strip())
                                    if "index" in data:
                                        completed_indices.add(data["index"])
                                        group_completed += 1
                        logger.info(
                            f"[Group {self.group_id}] Found {group_completed} completed samples from group {group_id} file: "
                            f"{group_result_file}"
                        )
                    except Exception as e:
                        logger.error(f"[Group {self.group_id}] Error reading existing file {group_result_file}: {e}")
        else:
            # 单group模式，只检查当前文件
            if osp.exists(self.result_file_path):
                try:
                    with open(self.result_file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line.strip())
                                if "index" in data:
                                    completed_indices.add(data["index"])
                    logger.info(
                        f"[Group {self.group_id}] Found {len(completed_indices)} completed samples in {self.result_file_path}"
                    )
                except Exception as e:
                    logger.error(f"[Group {self.group_id}] Error reading existing file {self.result_file_path}: {e}")

        logger.info(f"[Group {self.group_id}] Total completed samples across all groups: {len(completed_indices)}")
        return completed_indices

    def _post_process(self, response):
        ret = {"original_response": response}
        # 去掉思维链
        think_end = response.rfind("</think>")
        if think_end != -1:
            think_end += len("</think>")
            response = response[think_end:].strip()

        # 从box中提取答案
        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None and end != lt:
                response = resp[:end]
            else:
                response = "None"

        ret["prediction"] = response
        return ret

    def inference(self):
        if self.eval_only:
            logger.warning(f"[Group {self.group_id}] Cannot run inference in eval_only mode")
            return

        logger.info(f"Start running {self.model_name} x {self.dataset_name}")

        # 如果最终文件已存在且开启resume，直接退出
        if self.resume and osp.exists(self.final_result_file_path):
            logger.info(
                f"[Group {self.group_id}] Final result file {self.final_result_file_path} already exists, skipping inference"
            )
            return

        data_start = None
        duration_data = 0.0

        for idx in tqdm(range(len(self.dataloader)), desc=f"Group {self.group_id} processing {self.dataset_name}"):
            data_start = time.time()
            
            # if 'vllm_outputs' in locals(): del vllm_outputs
            # if 'outputs' in locals(): del outputs
            # if 'batch_inputs' in locals(): del batch_inputs

            batch = next(self.dataloader_iter)
            batch_line = batch["line"]
            batch_inputs = batch["inputs"]

            # 打印当前 batch 的视频路径
            try:
                for i, line in enumerate(batch_line):
                    if "video_path" in line:
                        logger.info(f"[OOM Debug] Processing video: {line['video_path']} (index={line.get('index', 'N/A')})")
                    else:
                        # 通用 fallback：打印整行或前几个字段
                        logger.info(f"[OOM Debug] Processing sample index={line.get('index', 'N/A')}, keys={list(line.keys())}")
            except Exception as e:
                logger.warning(f"[OOM Debug] Failed to log video path: {e}")

            # GPU推理部分
            infer_start = time.time()
            if self.use_vllm:
                vllm_outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params, use_tqdm=False)
                outputs = [o.outputs[0].text for o in vllm_outputs]

                if self.dataset_name in ["Charades-STA", "MMVU", "VideoMMMU"]:
                    batch_logprobs = [None for _ in vllm_outputs]
                else:
                    # -----------------------------------------------
                    # 提取 ABCD 的 logprobs
                    batch_logprobs = []
                    for o in vllm_outputs:
                        # 获取第一个 token 的 Top-20 概率字典 {token_id: logprob, ...}
                        if o.outputs[0].logprobs:
                            top_logprobs = o.outputs[0].logprobs[0]  # 这是一个字典

                            sample_probs = {}
                            # 遍历我们需要的目标（A, B, C, D）
                            for opt_char, opt_ids in self.option_tokens.items():
                                best_logprob = -999.0  # 默认极小值
                                found = False

                                # 在 Top-20 里找这些 ID
                                for tid in opt_ids:
                                    if tid in top_logprobs:
                                        # 如果找到了，取其中概率最大的（比如 "A" 和 " A" 哪个概率大取哪个）
                                        if top_logprobs[tid].logprob > best_logprob:
                                            best_logprob = top_logprobs[tid].logprob
                                            found = True

                                # 存入结果，如果 Top-20 里没找到，就存 None 或者 -999
                                sample_probs[opt_char] = best_logprob if found else None

                            batch_logprobs.append(sample_probs)
                        else:
                            batch_logprobs.append(None)
                    # -----------------------------------------------
            else:
                # transformers 将batch转成单个数据处理
                outputs = []
                # 新增：初始化一个空的 logprobs 列表，防止下面 crash
                batch_logprobs = []
                for inputs in batch_inputs:
                    inputs = inputs.to(self.model.device)
                    generated_ids = self.model.generate(**inputs, **self.generation_config)
                    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    outputs.append(output_text)
                    batch_logprobs.append(None)
            infer_end = time.time()
            duration_infer = infer_end - infer_start

            # CPU后处理部分
            post_start = time.time()
            batch_ret = [self._post_process(o) for o in outputs]

            for line, ret, logprob_dict in zip(batch_line, batch_ret, batch_logprobs):
                line["original_response"] = ret["original_response"]
                line["prediction"] = ret["prediction"]
                line["option_logprobs"] = logprob_dict  # 保存置信度

                # # 立即增量写入jsonl文件
                with open(self.result_file_path, "a", encoding="utf-8") as f:
                    json.dump(line.to_dict() if hasattr(line, "to_dict") else dict(line), f, ensure_ascii=False)
                    f.write("\n")
                # ---- 写入 jsonl：强制 index 在最前 ----
                # record = line.to_dict() if hasattr(line, 'to_dict') else dict(line)
                #
                # # 强制把 index 放在最前面（并把 numpy/int64 转成原生 int）
                # if 'index' in record:
                #     idx_val = record.pop('index')
                #     try:
                #         idx_val = int(idx_val)
                #     except Exception:
                #         pass
                #     record = {'index': idx_val, **record}
                #
                # with open(self.result_file_path, 'a', encoding='utf-8') as f:
                #     json.dump(record, f, ensure_ascii=False)
                #     f.write('\n')
                # -------------------------------------

            post_end = time.time()
            duration_post = post_end - post_start

            del_start = time.time()
            # 在循环结束前
            # del outputs
            if 'outputs' in locals(): del outputs
            # del batch_inputs
            if 'batch_inputs' in locals(): del batch_inputs
            if self.use_vllm:
                # del vllm_outputs
                if 'vllm_outputs' in locals(): del vllm_outputs
                if self.dataset_name in ["VideoMMMU", "MLVU", "LVBench", "Video-MME"]:
                    try:
                        self.llm.reset_mm_cache()
                        self.llm.reset_prefix_cache()
                    except Exception as e:
                        logger.warning(f"reset_mm_cache or reset_prefix_cache failed: {e}")
                if self.dataset_name in ["VideoMMMU","MLVU"]:
                    try:
                        if hasattr(self.llm, "sleep"):
                            ret = self.llm.sleep(level=2)  # level=2 最彻底
                            # 有的实现是 async
                            import inspect, asyncio
                            if inspect.iscoroutine(ret):
                                asyncio.run(ret)

                        if hasattr(self.llm, "wake_up"):
                            ret = self.llm.wake_up()
                            import inspect, asyncio
                            if inspect.iscoroutine(ret):
                                asyncio.run(ret)
                    except Exception as e:
                        logger.warning(f"sleep/wake_up failed: {e}")
            if self.dataset_name in ["VideoMMMU", "MLVU", "LVBench", "Video-MME"]:
                gc.collect() 
                torch.cuda.empty_cache()
            else:
                gc.collect() 
                if idx % 5 == 0:
                    torch.cuda.empty_cache()
                # 针对多进程 spawn 模式的额外清理 (对应 VLLM_WORKER_MULTIPROC_METHOD='spawn')
                # try:
                #     torch.cuda.ipc_collect()
                # except:
                #     pass
            logger.info(f"[Group {self.group_id}] Inference finished, results saved to {self.result_file_path}")

            del_end = time.time()
            duration_del = del_end - del_start

            data_end = time.time()
            duration_data = data_end - data_start

            logger.info(f"[Group {self.group_id}] Data: {duration_data:.2f}s, Inference {duration_infer:.2f}s, Post {duration_post:.2f}s, Del {duration_del:.2f}s")

    def merge_results(self):
        """合并所有进程的结果（仅由主进程调用）"""

        if self.num_groups <= 1:
            logger.info("Single process mode, no need to merge")
            return True

        logger.info("Starting to merge results from all groups...")

        # 加载数据集以获取期望的总长度
        try:
            logger.info("Loading dataset to validate result count...")
            # 创建一个临时的processor用于加载数据集
            from modelscope import AutoProcessor

            temp_processor = AutoProcessor.from_pretrained(self.model_path)
            temp_dataset = self.VIDEO_DATASET_CLS(temp_processor, group_id=0, num_groups=1, use_vllm=False, fps=2)
            expected_total_count = len(temp_dataset)
            logger.info(f"Expected total dataset length: {expected_total_count}")
        except Exception as e:
            logger.error(f"Failed to load dataset for validation: {e}")
            return False

        # 合并所有rank的结果文件
        expected_files = []
        for group_id in range(self.num_groups):
            expected_file = osp.join(
                self.pred_root, f"{group_id}{self.num_groups}_{self.model_name}_{self.dataset_name}.jsonl"
            )
            expected_files.append(expected_file)

        # 检查所有文件是否存在
        missing_files = [f for f in expected_files if not osp.exists(f)]
        if missing_files:
            logger.error(f"Missing result files: {missing_files}")
            return False

        # 合并所有结果
        all_results = []
        for rank_file in expected_files:
            try:
                rank_data = []
                with open(rank_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rank_data.append(json.loads(line.strip()))
                all_results.extend(rank_data)
                logger.info(f"Loaded {len(rank_data)} results from {rank_file}")
            except Exception as e:
                logger.error(f"Error loading {rank_file}: {e}")
                return False

        if all_results:
            # 校验合并后的结果数量是否与数据集长度一致
            actual_count = len(all_results)
            if actual_count != expected_total_count:
                logger.warning(f"Result count mismatch! Expected: {expected_total_count}, Actual: {actual_count}")
                logger.warning("Skipping file merge and cleanup due to incomplete results")
                return False

            logger.info(f"Result count validation passed: {actual_count} results match dataset length")

            # 转换为DataFrame并按index排序
            merged_df = pd.DataFrame(all_results)
            if "index" in merged_df.columns:
                merged_df = merged_df.sort_values("index").reset_index(drop=True)
                # 把 index 移到第一列
                # cols = ['index'] + [c for c in merged_df.columns if c != 'index']
                # merged_df = merged_df[cols]

            merged_df.to_excel(self.final_result_file_path, index=False)
            logger.info(
                f"Merged results saved to {self.final_result_file_path} with {len(merged_df)} samples (sorted by index)"
            )

            # 清理临时文件
            for rank_file in expected_files:
                try:
                    # os.remove(rank_file)
                    logger.info(f"Removed temporary file {rank_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {rank_file}: {e}")
            return True
        else:
            logger.error("No results to merge")
            return False

    def evaluate(self):
        """执行评估"""
        if self.num_groups > 1:
            eval_file = self.final_result_file_path
        else:
            eval_file = self.result_file_path

        if not osp.exists(eval_file):
            logger.error(f"Evaluation file not found: {eval_file}")
            return

        logger.info(f"Starting evaluation on {eval_file}")
        judge_kwargs = dict(model="qwen__qwen3-vl-235b-a22b-instruct", nproc=4)
        rating = self.VIDEO_DATASET_CLS.evaluate(eval_file, **judge_kwargs)
        logger.info(json.dumps(rating, ensure_ascii=False, indent=2))


def worker(group_id, num_groups, args):
    logger.info(f"[Group {group_id}] Process initialized, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"Using vLLM: {args.use_vllm}")  # 调试用日志

    evaluator = Evaluator(
        group_id=group_id,
        num_groups=num_groups,
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output,
        moe=args.moe,
        use_vllm=args.use_vllm,
        post_process=args.post_process,
        resume=not args.disable_resume,
        eval_only=False,  # Worker需要完整初始化
        gpu_memory_utilization=args.gpu_memory_utilization,
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

        # 创建环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env["WORLD_SIZE"] = str(world_size)
        env["VLLM_HOST_IP"] = "127.0.0.1"
        env["VLLM_PORT"] = str(_find_free_port())

        logger.info(
            f"[Group {group_id}] WORLD_SIZE={world_size}, CUDA_VISIBLE_DEVICES={gpu_ids}, "
            f"{env['VLLM_HOST_IP']}:{env['VLLM_PORT']}"
        )

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
        cmd.extend(["--gpu_memory_utilization", str(args.gpu_memory_utilization)])

        if args.moe:
            cmd.append("--moe")
        if args.use_vllm:
            cmd.append("--use_vllm")
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
        use_vllm=args.use_vllm,
        post_process=args.post_process,
        resume=not args.disable_resume,
        eval_only=True,  # 使用eval_only模式，不初始化模型和数据
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
    parser.add_argument("--worker", action="store_true", help="Worker mode")    # 是否为worker模式
    parser.add_argument("--group_id", type=int, default=0)                      # 组id
    parser.add_argument("--num_groups", type=int, default=1)                    # 组数
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct") # 模型名称
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct") # 模型路径
    parser.add_argument("--dataset", type=str, default="Video-MME")              # 数据集名称
    parser.add_argument("--output", type=str, default="./outputs/Qwen3-VL")      # 输出路径
    parser.add_argument("--moe", action="store_true", default=False)             # 是否使用moe
    parser.add_argument("--use_vllm", action="store_true", default=False)        # 是否使用vllm
    parser.add_argument("--post_process", action="store_true", default=False)    # 处理cot输出
    parser.add_argument("--disable_resume", action="store_true", default=False)  # 是否禁用续写
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)     # gpu内存利用率

    args = parser.parse_args()

    if args.worker:
        # Worker模式：执行实际任务
        worker(args.group_id, args.num_groups, args)
    else:
        # Main模式：启动多个worker
        main(args.nproc)
