import json
import logging
import os
import os.path as osp
import time

import pandas as pd
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm import tqdm
from vlmeval.smp import githash, timestr

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
        use_vllm=True,
        post_process=False,
        resume=False,
        eval_only=False,
        gpu_memory_utilization=0.5,
        sample_path=None,
        base_url="http://127.0.0.1:22002/v1",
        api_key="EMPTY",
        request_timeout=3600,
        max_retries=3,
        fps=2,
    ):
        del moe, use_vllm, gpu_memory_utilization, sample_path

        self.group_id = group_id
        self.num_groups = num_groups
        self.model_name = model_name
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.resume = resume
        self.eval_only = eval_only
        self.post_process = post_process
        self.base_url = base_url
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.fps = fps

        date, commit_id = timestr("day"), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        self.pred_root = osp.join(output_dir, self.model_name, self.dataset_name, eval_id)
        if self.num_groups > 1:
            self.result_file_path = osp.join(
                self.pred_root, f"{self.group_id}{self.num_groups}_{self.model_name}_{self.dataset_name}.jsonl"
            )
        else:
            self.result_file_path = osp.join(self.pred_root, f"{self.model_name}_{self.dataset_name}.jsonl")
        self.final_result_file_path = osp.join(self.pred_root, f"{self.model_name}_{self.dataset_name}.xlsx")
        os.makedirs(self.pred_root, exist_ok=True)

        self.VIDEO_DATASET_CLS = self._resolve_dataset_cls(dataset_name)

        if eval_only:
            logger.info("[Group %s] eval_only=True, skip dataset and client init", self.group_id)
            self.dataloader = None
            self.dataloader_iter = None
            self.client = None
            return

        completed_indices = self._get_completed_indices()
        dataset = self.VIDEO_DATASET_CLS(
            None,
            group_id=group_id,
            num_groups=num_groups,
            fps=self.fps,
            completed_indices=completed_indices,
        )
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=0,
        )
        self.dataloader_iter = iter(self.dataloader)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.request_timeout, max_retries=0)
        self.max_tokens = self._resolve_max_tokens(dataset_name)

    @staticmethod
    def _resolve_dataset_cls(dataset_name):
        if dataset_name == "Video-MME":
            return VideoMMEDataset
        if dataset_name == "MVBench":
            return MVBenchDataset
        if dataset_name == "MLVU":
            return MLVUDataset
        if dataset_name == "LVBench":
            return LVBenchDataset
        if dataset_name == "Charades-STA":
            return CharadesSTADataset
        if dataset_name == "MMVU":
            return MMVUDataset
        if dataset_name == "VideoMMMU":
            return VideoMMMUDataset
        if dataset_name == "FCMBench":
            return FCMBenchDataset
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    @staticmethod
    def _resolve_max_tokens(dataset_name):
        if dataset_name == "Charades-STA":
            return 32
        if dataset_name in ["MMVU", "VideoMMMU"]:
            return 32768
        if dataset_name == "FCMBench":
            return 1000
        return 10

    def _get_completed_indices(self):
        completed_indices = set()
        if not self.resume:
            return completed_indices

        if self.num_groups > 1:
            for group_id in range(self.num_groups):
                group_result_file = osp.join(
                    self.pred_root, f"{group_id}{self.num_groups}_{self.model_name}_{self.dataset_name}.jsonl"
                )
                if not osp.exists(group_result_file):
                    continue
                with open(group_result_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line.strip())
                        if "index" in data:
                            completed_indices.add(data["index"])
        elif osp.exists(self.result_file_path):
            with open(self.result_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    if "index" in data:
                        completed_indices.add(data["index"])

        logger.info("[Group %s] resume found %s completed samples", self.group_id, len(completed_indices))
        return completed_indices

    def _post_process(self, response):
        ret = {"original_response": response}
        think_end = response.rfind("</think>")
        if think_end != -1:
            response = response[think_end + len("</think>") :].strip()

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
                if i == lt - 1:
                    end = lt
                    break
            response = resp[:end] if end is not None and end != lt else "None"

        ret["prediction"] = response
        return ret

    def _chat_once(self, request_payload):
        return self.client.chat.completions.create(
            model=self.model_path,
            messages=request_payload["messages"],
            max_tokens=self.max_tokens,
            extra_body=request_payload.get("extra_body", {}),
        )

    def _chat_with_retry(self, request_payload):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._chat_once(request_payload)
            except Exception as e:
                last_err = e
                logger.warning(
                    "[Group %s] request failed (%s/%s): %s",
                    self.group_id,
                    attempt,
                    self.max_retries,
                    e,
                )
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"Request failed after {self.max_retries} retries: {last_err}")

    def inference(self):
        if self.eval_only:
            logger.warning("[Group %s] Cannot run inference in eval_only mode", self.group_id)
            return

        if self.resume and osp.exists(self.final_result_file_path):
            logger.info("[Group %s] final xlsx exists, skip inference", self.group_id)
            return

        logger.info(
            "[Group %s] start %s on %s via %s",
            self.group_id,
            self.model_name,
            self.dataset_name,
            self.base_url,
        )

        for _ in tqdm(range(len(self.dataloader)), desc=f"Group {self.group_id} processing {self.dataset_name}"):
            batch = next(self.dataloader_iter)
            batch_line = batch["line"]
            batch_inputs = batch["inputs"]

            outputs = []
            for req in batch_inputs:
                resp = self._chat_with_retry(req)
                content = resp.choices[0].message.content
                if isinstance(content, list):
                    content = "\n".join(
                        [c.get("text", "") for c in content if isinstance(c, dict)]
                    )
                outputs.append(content or "")

            batch_ret = [self._post_process(o) for o in outputs]
            for line, ret in zip(batch_line, batch_ret):
                line["original_response"] = ret["original_response"]
                line["prediction"] = ret["prediction"]
                line["option_logprobs"] = None
                with open(self.result_file_path, "a", encoding="utf-8") as f:
                    json.dump(line.to_dict() if hasattr(line, "to_dict") else dict(line), f, ensure_ascii=False)
                    f.write("\n")

        logger.info("[Group %s] inference done, saved to %s", self.group_id, self.result_file_path)

    def merge_results(self):
        if self.num_groups <= 1:
            return True

        expected_files = [
            osp.join(self.pred_root, f"{group_id}{self.num_groups}_{self.model_name}_{self.dataset_name}.jsonl")
            for group_id in range(self.num_groups)
        ]

        for file_path in expected_files:
            if not osp.exists(file_path):
                logger.error("Missing result file: %s", file_path)
                return False

        temp_dataset = self.VIDEO_DATASET_CLS(None, group_id=0, num_groups=1, fps=self.fps)
        expected_total_count = len(temp_dataset)

        all_results = []
        for file_path in expected_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line.strip()))

        if len(all_results) != expected_total_count:
            logger.error("Result count mismatch: expected %s, got %s", expected_total_count, len(all_results))
            return False

        merged_df = pd.DataFrame(all_results)
        if "index" in merged_df.columns:
            merged_df = merged_df.sort_values("index").reset_index(drop=True)
        merged_df.to_excel(self.final_result_file_path, index=False)
        logger.info("Merged results saved to %s", self.final_result_file_path)
        return True

    def evaluate(self):
        return None
