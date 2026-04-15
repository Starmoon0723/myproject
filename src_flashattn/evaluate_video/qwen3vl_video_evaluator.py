import json
import logging
import os
import os.path as osp
import time

import pandas as pd
import torch
from openai import OpenAI
from modelscope import AutoProcessor, Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import subprocess
# from vlmeval.smp import githash, timestr

from datasets_video import (
    FCMBenchDataset,
    CharadesSTADataset,
    LVBenchDataset,
    MLVUDataset,
    MMVUDataset,
    MVBenchDataset,
    VideoMMEDataset,
    VideoMMMUDataset,
    collate_fn,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def timestr(format_type="day"):
    """
    获取当前时间字符串
    format_type: "day" -> YYYYMMDD, "hour" -> YYYYMMDD_HHMMSS
    """
    now = datetime.now()
    if format_type == "day":
        return now.strftime("%Y%m%d")
    elif format_type == "hour":
        return now.strftime("%Y%m%d_%H%M%S")
    else:
        return now.strftime("%Y%m%d")

def githash(digits=8):
    """
    获取当前 git 仓库的 commit hash
    digits: 返回的哈希位数
    """
    try:
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 执行 git 命令获取 commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=current_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return result.stdout.strip()[:digits]
        else:
            return "unknown"
    except Exception:
        return "unknown"


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
        post_process=False,
        resume=False,
        eval_only=False,
        gpu_memory_utilization=0.5,
        sample_path=None,
    ):
        self.group_id = group_id
        self.num_groups = num_groups
        self.model_name = model_name
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.resume = resume
        self.eval_only = eval_only
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sample_path = sample_path

        if self.group_id == 0:
            logger.info(f"Model name: {self.model_name}")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Dataset name: {self.dataset_name}")

        date, commit_id = timestr("day"), githash(digits=8)  # noqa: F405
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
        elif dataset_name == "FCMBench":
            self.VIDEO_DATASET_CLS = FCMBenchDataset
        else:
            raise ValueError("鍙敮鎸乂ideo-MME銆丮VBench銆丮LVU銆丩VBench銆丆harades-STA銆丮MVU銆乂ideoMMMU銆丗CMBench")

        # 如果是eval_only模式，跳过所有耗时的初始化
        if eval_only:
            logger.info(f"[Group {self.group_id}] Running in eval_only mode, skipping model and data initialization")
            self.processor = None
            self.dataloader = None
            self.dataloader_iter = None
            self.model = None
            return

        # 以下是正常模式的初始化        
        self.processor = AutoProcessor.from_pretrained(model_path)

        # 获取已完成的数据索引
        completed_indices = self._get_completed_indices()

        dataset = self.VIDEO_DATASET_CLS(
            self.processor,
            group_id=group_id,
            num_groups=num_groups,
            fps=2,
            # nframe=64,
            completed_indices=completed_indices,
            sample_path=self.sample_path,
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
            multiprocessing_context="spawn",
        )
        # 创建迭代器预热数据        
        self.dataloader_iter = iter(self.dataloader)

        logger.info(f"Initializing model from {self.model_path}")
        logger.info("Im using flash-attn")
        MODEL_CLS = Qwen3VLMoeForConditionalGeneration if moe else Qwen3VLForConditionalGeneration
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
                attn_implementation="sdpa",
            )
        self.generation_config = dict(
            do_sample=True,
            top_p=0.8,
            top_k=20,
            temperature=0.7,
            repetition_penalty=1.0,
            max_new_tokens=10,
        )
        if self.dataset_name == "Charades-STA":
            self.generation_config["max_new_tokens"] = 32
        elif self.dataset_name == "MMVU":
            self.generation_config["max_new_tokens"] = 32768
        elif self.dataset_name == "VideoMMMU":
            self.generation_config["max_new_tokens"] = 32768
        elif self.dataset_name == "FCMBench":
            self.generation_config["max_new_tokens"] = 1000
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

        for idx in tqdm(range(len(self.dataloader)), desc=f"Group {self.group_id} processing {self.dataset_name}"):
            data_start = time.time()

            batch = next(self.dataloader_iter)
            batch_line = batch["line"]
            batch_inputs = batch["inputs"]

            # 打印当前 batch 的视频路径         
            try:
                for line in batch_line:
                    if "video_path" in line:
                        logger.info(
                            f"[OOM Debug] Processing video: {line['video_path']} (index={line.get('index', 'N/A')})"
                        )
                    else:
                        logger.info(
                            f"[OOM Debug] Processing sample index={line.get('index', 'N/A')}, keys={list(line.keys())}"
                        )
            except Exception as e:
                logger.warning(f"[OOM Debug] Failed to log video path: {e}")

            # GPU推理部分
            infer_start = time.time()
            outputs = []
            batch_logprobs = []
            for inputs in batch_inputs:
                inputs = inputs.to(self.model.device)
                generated_ids = self.model.generate(**inputs, **self.generation_config)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                outputs.append(output_text)
                batch_logprobs.append(None)
            infer_end = time.time()
            
            # CPU后处理部分            
            post_start = time.time()
            batch_ret = [self._post_process(o) for o in outputs]

            for line, ret, logprob_dict in zip(batch_line, batch_ret, batch_logprobs):
                line["original_response"] = ret["original_response"]
                line["prediction"] = ret["prediction"]
                line["option_logprobs"] = logprob_dict
                # 立即增量写入jsonl文件
                with open(self.result_file_path, "a", encoding="utf-8") as f:
                    json.dump(line.to_dict() if hasattr(line, "to_dict") else dict(line), f, ensure_ascii=False)
                    f.write("\n")

            post_end = time.time()
            duration_infer = infer_end - infer_start
            duration_post = post_end - post_start
            duration_data = post_end - data_start

            logger.info(
                f"[Group {self.group_id}] Data: {duration_data:.2f}s, Inference {duration_infer:.2f}s, "
                f"Post {duration_post:.2f}s"
            )

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
        # if self.num_groups > 1:
        #     eval_file = self.final_result_file_path
        # else:
        #     eval_file = self.result_file_path

        # if not osp.exists(eval_file):
        #     logger.error(f"Evaluation file not found: {eval_file}")
        #     return

        # logger.info(f"Starting evaluation on {eval_file}")
        # judge_kwargs = dict(model="qwen__qwen3-vl-235b-a22b-instruct", nproc=4)
        # rating = self.VIDEO_DATASET_CLS.evaluate(eval_file, **judge_kwargs)
        # logger.info(json.dumps(rating, ensure_ascii=False, indent=2))

