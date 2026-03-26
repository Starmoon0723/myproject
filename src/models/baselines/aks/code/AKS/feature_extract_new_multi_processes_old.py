import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

import numpy as np
import torch
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval

# Note: keep imports at top-level for compatibility with existing environment.
# We use multiprocessing "spawn" (Windows-safe) and ensure CUDA device is set per worker.
import multiprocessing as mp


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_name', type=str, default='longvideobench', help='support longvideobench and videomme')
    parser.add_argument('--label_path', type=str, default='./datasets/longvideobench/lvb_val.json',help='your path of the label')
    parser.add_argument('--video_path', type=str, default='./datasets/longvideobench/videos',help='your path of the video')
    parser.add_argument('--extract_feature_model', type=str,default='blip', help='blip/clip/sevila')
    parser.add_argument('--output_file', type=str,default='./outscores',help='path of output scores and frames')
    parser.add_argument('--device', type=str,default='cuda')
    parser.add_argument('--blip_model_path', type=str, default='', help='local path for BLIP (e.g., ./models/blip-itm-large-coco)')
    parser.add_argument('--clip_model_path', type=str, default='', help='local path for CLIP (e.g., ./models/clip-vit-base-patch32)')

    # --- Parallel / Multi-GPU (new, default keeps original single-process behavior) ---
    parser.add_argument('--num_workers', type=int, default=1, help='number of worker processes (1 keeps original sequential behavior)')
    parser.add_argument('--gpu_ids', type=str, default='', help='comma-separated GPU ids to use, e.g. "0,1". If empty, will use current visible GPUs.')
    parser.add_argument('--workers_per_gpu', type=int, default=1, help='how many model copies (processes) to run per GPU id')
    parser.add_argument('--chunk_size', type=int, default=32, help='number of videos per chunk/part file per worker assignment')
    parser.add_argument('--part_dir', type=str, default='', help='optional custom directory for part files (default: <out_score_path>/parts)')

    return parser.parse_args()


@dataclass(frozen=True)
class _ResolvedPaths:
    label_path: str
    video_root: str
    out_score_path: str
    score_file_path: str
    frame_file_path: str


def _resolve_paths(args) -> _ResolvedPaths:
    # --- Path Setup ---
    # videomme, mvbench, lvbench, mlvu, mmvu, videommmu
    if args.dataset_name == "longvideobench":
        label_path = os.path.join(args.label_path, 'lvb_val.json')
        video_root = os.path.join(args.video_path, 'videos')
    elif args.dataset_name == "videomme":
        label_path = os.path.join(args.label_path, 'videomme.json')
        video_root = os.path.join(args.video_path, 'data')
    elif args.dataset_name == "mlvu":
        label_path = os.path.join(args.label_path, 'mlvu_all.json')
        video_root = args.video_path
    elif args.dataset_name == "mmvu":
        label_path = os.path.join(args.label_path, 'validation_process.json')
        video_root = args.video_path
    elif args.dataset_name == "videommmu":
        label_path = os.path.join(args.label_path, 'VideoMMMU.tsv')  # 注意：这里是 .tsv
        video_root = args.video_path
    elif args.dataset_name == "lvbench":
        # label_path = os.path.join(args.label_path, 'video_info.meta.json')
        label_path = os.path.join(args.label_path, 'video_info.meta_wo_options.json')
        video_root = args.video_path
    elif args.dataset_name == "mvbench":
        label_path = os.path.join(args.label_path, 'mvbench_all.json')
        video_root = args.video_path
    elif args.dataset_name == "fcmbench":
        label_path = os.path.join(args.label_path, 'fcmbench_longvideo_v1.0_20260205_absolute.jsonl')
        video_root = args.video_path
    else:
        raise ValueError("dataset_name: longvideobench or videomme or mlvu or mmvu or videommmu or lvbench or mvbench or fcmbench")

    # --- Output Directory Setup ---
    dataset_out_dir = os.path.join(args.output_file, args.dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)
    out_score_path = os.path.join(dataset_out_dir, args.extract_feature_model)
    os.makedirs(out_score_path, exist_ok=True)

    score_file_path = os.path.join(out_score_path, 'scores.json')
    frame_file_path = os.path.join(out_score_path, 'frames.json')

    return _ResolvedPaths(
        label_path=label_path,
        video_root=video_root,
        out_score_path=out_score_path,
        score_file_path=score_file_path,
        frame_file_path=frame_file_path,
    )


def _load_datas(label_path: str):
    # if os.path.exists(label_path):
    #     with open(label_path, 'r', encoding='utf-8') as f:
    #         return json.load(f)
    # raise OSError("the label file does not exist")
    if not os.path.exists(label_path):
        raise OSError(f"The label file does not exist: {label_path}")

    if label_path.endswith('.tsv'):
        # Load VideoMMMU TSV
        df = pd.read_csv(label_path, sep='\t')
        # Convert to list of dicts (like JSON)
        datas = df.to_dict(orient='records')
        # Optional: clean up <image N> tags in question
        # for d in datas:
        #     if 'question' in d and isinstance(d['question'], str):
        #         # Remove <image 1>, <image 2>, etc.
        #         import re
        #         d['question'] = re.sub(r'<image \d+>', '', d['question']).strip()
        return datas
    elif label_path.endswith('.jsonl'):
        with open(label_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        # Original JSON loading
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def _count_existing_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def _build_model_and_processors(args, device: str):
    # --- Model Loading (Unchanged) ---
    use_local_blip = False
    blip_processor = None
    vis_processors = None
    text_processors = None
    processor = None

    if args.extract_feature_model == 'blip':
        blip_path = os.path.join(args.blip_model_path, "blip-itm-large-coco-ssd")
        if blip_path and os.path.exists(blip_path):
            use_local_blip = True
            print(f"Loading local BLIP from {blip_path}")
            model = BlipForImageTextRetrieval.from_pretrained(blip_path, local_files_only=True).to(device)
            blip_processor = BlipProcessor.from_pretrained(blip_path, local_files_only=True)
        else:
            print("Loading BLIP from LAVIS hub...")
            model, vis_processors, text_processors = load_model_and_preprocess(
                "blip_image_text_matching", "large", device=device, is_eval=True
            )
    elif args.extract_feature_model == 'clip':
        clip_path = os.path.join(args.clip_model_path, "clip-vit-base-patch32-ssd")
        model = CLIPModel.from_pretrained(clip_path or "openai/clip-vit-base-patch32", local_files_only=bool(clip_path))
        model.to(device)
        processor = CLIPProcessor.from_pretrained(clip_path or "openai/clip-vit-base-patch32", local_files_only=bool(clip_path))
    else:
        raise ValueError("model not support")

    return model, processor, blip_processor, vis_processors, text_processors, use_local_blip


def _video_path_for_data(dataset_name: str, video_root: str, data: dict) -> str:
    if dataset_name == 'longvideobench':
        return os.path.join(video_root, data["video_path"])
    if dataset_name == 'videomme':
        return os.path.join(video_root, data["videoID"] + '.mp4')
    if dataset_name == 'mlvu':
        return data["video_path"]
    if dataset_name == 'mmvu':
        return data["video_path"]
    if dataset_name == 'videommmu':
        return data["video_path"]
    if dataset_name == 'lvbench':
        return data["video_path"]
    if dataset_name == 'mvbench':
        return data["video_path"]
    if dataset_name == 'fcmbench':
        return data["video_path"]
    raise ValueError("dataset_name: longvideobench or videomme or mlvu or mmvu or videommmu or lvbench or mvbench or fcmbench")

# def _process_one_item(
#     *,
#     args,
#     dataset_name: str,
#     video_root: str,
#     data: dict,
#     device: str,
#     model,
#     processor,
#     blip_processor,
#     vis_processors,
#     text_processors,
#     use_local_blip: bool,
# ):
#     text = data['question']
#     video = _video_path_for_data(dataset_name, video_root, data)

#     try:
#         if not os.path.exists(video):
#             print(f"Warning: Video file not found: {video}")
#             return [], [], video
        
#         # 仍然使用 CPU 解码，但增加 num_threads 提升单进程速度
#         vr = VideoReader(video, ctx=cpu(0), num_threads=2)
#         fps = vr.get_avg_fps()
#         # 确定需要抽取的帧索引
#         frame_indices = [j * int(fps) for j in range(int(len(vr) / int(fps)))]
        
#         if not frame_indices:
#             return [], [], video

#         score = []
#         frame_num = frame_indices.copy()

#         # --- 核心优化：Batch 处理 ---
#         if args.extract_feature_model == 'clip':
#             # 1. 一次性从视频读取所有需要的帧 (CPU)
#             frames = vr.get_batch(frame_indices).asnumpy() # shape: [N, H, W, C]
#             pil_images = [Image.fromarray(f) for f in frames]

#             # 2. 文本特征预计算 (GPU)
#             inputs_text = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
#             with torch.no_grad():
#                 text_features = model.get_text_features(**inputs_text) # [1, D]

#             # 3. 图像特征批量计算 (GPU)
#             # 将所有图片一次性交给 processor 处理
#             inputs_images = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
#             with torch.no_grad():
#                 image_features = model.get_image_features(**inputs_images) # [N, D]
            
#             # 4. 批量计算余弦相似度
#             logits = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
#             score = logits.cpu().tolist()

#         elif args.extract_feature_model == 'blip':
#             # BLIP 的 ITM 模块通常需要图像和文本一一配对，Batch 逻辑稍复杂
#             # 这里建议至少将图像预处理批量化
#             frames = vr.get_batch(frame_indices).asnumpy()
#             for f in frames:
#                 raw_image = Image.fromarray(f)
#                 # ... 保持你原有的 BLIP 处理逻辑 ...
#                 # (BLIP 逻辑通常较重，如果 CPU 瓶颈严重，建议优先用 CLIP)

#     except Exception as e:
#         print(f"Error processing video {video}: {e}")
#         score, frame_num = [], []

#     return score, frame_num, video
    
def _process_one_item(
    *,
    args,
    dataset_name: str,
    video_root: str,
    data: dict,
    device: str,
    model,
    processor,
    blip_processor,
    vis_processors,
    text_processors,
    use_local_blip: bool,
):
    # --- Data Preparation (Unchanged) ---
    try:
        text = data['question']
    except:
        text = data['prompt']
    video = _video_path_for_data(dataset_name, video_root, data)

    try:
        # Add basic error handling for bad video files so script doesn't die
        if not os.path.exists(video):
            print(f"Warning: Video file not found: {video}")
            score = []
            frame_num = []
        else:
            duration = data.get('duration', None)  # keep original semantics (unused)
            # gpu_id = int(device.split(':')[-1]) if 'cuda' in device else 0
            vr = VideoReader(video, ctx=cpu(0), num_threads=2)
            # vr = VideoReader(video, ctx=gpu(gpu_id), num_threads=2)
            fps = vr.get_avg_fps()
            frame_nums = int(len(vr) / int(fps))

            score = []
            frame_num = []

            # --- Feature Extraction (Unchanged Logic) ---
            if args.extract_feature_model == 'blip':
                if use_local_blip:
                    for j in range(frame_nums):
                        raw_image = np.array(vr[j * int(fps)])
                        raw_image = Image.fromarray(raw_image)
                        inputs = blip_processor(images=raw_image, text=text, return_tensors="pt").to(device)
                        with torch.no_grad():
                            blip_output = model(**inputs, return_dict=True)
                        itm_scores = getattr(blip_output, "itm_scores", None)
                        if itm_scores is None:
                            itm_scores = getattr(blip_output, "logits_per_image", None)
                        if itm_scores is None:
                            itm_scores = blip_output[0]
                        blip_scores = torch.nn.functional.softmax(itm_scores, dim=1)
                        score.append(blip_scores[:, 1].item())
                        frame_num.append(j * int(fps))
                else:
                    txt = text_processors["eval"](text)
                    for j in range(frame_nums):
                        raw_image = np.array(vr[j * int(fps)])
                        raw_image = Image.fromarray(raw_image)
                        img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            blip_output = model({"image": img, "text_input": txt}, match_head="itm")
                        blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                        score.append(blip_scores[:, 1].item())
                        frame_num.append(j * int(fps))

            elif args.extract_feature_model == 'clip':
                inputs_text = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                text_features = model.get_text_features(**inputs_text)
                for j in range(frame_nums):
                    raw_image = np.array(vr[j * int(fps)])
                    raw_image = Image.fromarray(raw_image)
                    inputs_image = processor(images=raw_image, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs_image)
                    clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
                    score.append(clip_score.item())
                    frame_num.append(j * int(fps))
    except Exception as e:
        print(f"Error processing video: {e}")
        score = []
        frame_num = []

    return score, frame_num, video


def _parse_gpu_ids(gpu_ids: str) -> List[int]:
    gpu_ids = (gpu_ids or "").strip()
    if not gpu_ids:
        # Respect current visibility; try to use 0..N-1 of visible devices.
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        return list(range(n)) if n > 0 else []
    ids = []
    for part in gpu_ids.split(","):
        part = part.strip()
        if part == "":
            continue
        ids.append(int(part))
    return ids


def _worker_process_range(
    *,
    worker_id: int,
    gpu_id: Optional[int],
    start_idx: int,
    end_idx: int,
    args_dict: dict,
    resolved: _ResolvedPaths,
    part_dir: str,
):
    # Recreate args-like object
    class _ArgsObj:
        pass

    args = _ArgsObj()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # Set GPU device for this worker (supports multiple workers on same GPU)
    if args.device.startswith("cuda") and torch.cuda.is_available() and gpu_id is not None:
        try:
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"[worker {worker_id}] Warning: failed to set cuda device {gpu_id}: {e}")
        device = f"cuda:{gpu_id}"
    else:
        device = args.device

    datas = _load_datas(resolved.label_path)
    model, processor, blip_processor, vis_processors, text_processors, use_local_blip = _build_model_and_processors(args, device)

    score_part_path = os.path.join(part_dir, f"scores.part_{start_idx:09d}_{end_idx:09d}.jsonl")
    frame_part_path = os.path.join(part_dir, f"frames.part_{start_idx:09d}_{end_idx:09d}.jsonl")
    done_flag = os.path.join(part_dir, f"done.part_{start_idx:09d}_{end_idx:09d}.flag")

    # If already done (line count matches), skip
    expected = max(0, end_idx - start_idx)
    if os.path.exists(done_flag):
        if _count_existing_lines(score_part_path) == expected and _count_existing_lines(frame_part_path) == expected:
            print(f"[worker {worker_id}] Part {start_idx}-{end_idx} already done. Skipping.")
            return

    os.makedirs(part_dir, exist_ok=True)
    print(f"[worker {worker_id}] GPU={gpu_id} processing idx [{start_idx}, {end_idx}) -> parts")
    t0 = time.time()

    with open(score_part_path, 'w', encoding='utf-8') as f_score, open(frame_part_path, 'w', encoding='utf-8') as f_frame:
        for idx in range(start_idx, end_idx):
            data = datas[idx]
            score, frame_num, video = _process_one_item(
                args=args,
                dataset_name=args.dataset_name,
                video_root=resolved.video_root,
                data=data,
                device=device,
                model=model,
                processor=processor,
                blip_processor=blip_processor,
                vis_processors=vis_processors,
                text_processors=text_processors,
                use_local_blip=use_local_blip,
            )
            # Keep original line format: one JSON array per line
            f_score.write(json.dumps(score) + "\n")
            f_frame.write(json.dumps(frame_num) + "\n")

    # Mark done
    with open(done_flag, 'w', encoding='utf-8') as f:
        f.write("ok\n")
    print(f"[worker {worker_id}] Finished part {start_idx}-{end_idx} in {time.time() - t0:.1f}s")


def _worker_loop(
    *,
    worker_id: int,
    gpu_id: Optional[int],
    args_dict: dict,
    resolved: _ResolvedPaths,
    part_dir: str,
    task_queue,
):
    """
    Long-lived worker: load model once, then process many (start_idx, end_idx) tasks.
    This avoids repeated model loading overhead (critical for speed).
    """
    # Recreate args-like object
    class _ArgsObj:
        pass

    args = _ArgsObj()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # Set GPU device for this worker (supports multiple workers on same GPU)
    if args.device.startswith("cuda") and torch.cuda.is_available() and gpu_id is not None:
        try:
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"[worker {worker_id}] Warning: failed to set cuda device {gpu_id}: {e}")
        device = f"cuda:{gpu_id}"
    else:
        device = args.device

    datas = _load_datas(resolved.label_path)
    model, processor, blip_processor, vis_processors, text_processors, use_local_blip = _build_model_and_processors(args, device)

    print(f"[worker {worker_id}] Ready. GPU={gpu_id}, device={device}")
    while True:
        task = task_queue.get()
        if task is None:
            break
        start_idx, end_idx = task

        score_part_path = os.path.join(part_dir, f"scores.part_{start_idx:09d}_{end_idx:09d}.jsonl")
        frame_part_path = os.path.join(part_dir, f"frames.part_{start_idx:09d}_{end_idx:09d}.jsonl")
        done_flag = os.path.join(part_dir, f"done.part_{start_idx:09d}_{end_idx:09d}.flag")

        expected = max(0, end_idx - start_idx)
        if os.path.exists(done_flag):
            if _count_existing_lines(score_part_path) == expected and _count_existing_lines(frame_part_path) == expected:
                print(f"[worker {worker_id}] Part {start_idx}-{end_idx} already done. Skipping.")
                continue

        os.makedirs(part_dir, exist_ok=True)
        print(f"[worker {worker_id}] Processing idx [{start_idx}, {end_idx})")
        t0 = time.time()

        with open(score_part_path, 'w', encoding='utf-8') as f_score, open(frame_part_path, 'w', encoding='utf-8') as f_frame:
            for idx in range(start_idx, end_idx):
                data = datas[idx]
                score, frame_num, _ = _process_one_item(
                    args=args,
                    dataset_name=args.dataset_name,
                    video_root=resolved.video_root,
                    data=data,
                    device=device,
                    model=model,
                    processor=processor,
                    blip_processor=blip_processor,
                    vis_processors=vis_processors,
                    text_processors=text_processors,
                    use_local_blip=use_local_blip,
                )
                # Keep original line format: one JSON array per line
                f_score.write(json.dumps(score) + "\n")
                f_frame.write(json.dumps(frame_num) + "\n")

        with open(done_flag, 'w', encoding='utf-8') as f:
            f.write("ok\n")
        print(f"[worker {worker_id}] Finished part {start_idx}-{end_idx} in {time.time() - t0:.1f}s")


def _make_ranges(start: int, end: int, chunk_size: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0:
        chunk_size = 1
    ranges = []
    cur = start
    while cur < end:
        nxt = min(end, cur + chunk_size)
        ranges.append((cur, nxt))
        cur = nxt
    return ranges

def main(args):
    resolved = _resolve_paths(args)
    datas = _load_datas(resolved.label_path)
    total_videos = len(datas)

    # --- Resume Logic (keep original meaning: count existing lines) ---
    processed_count = _count_existing_lines(resolved.score_file_path)
    if processed_count > 0:
        print(f"Found existing output file. {processed_count} videos already processed. Resuming...")
    if processed_count >= total_videos:
        print("Processing complete. Nothing to do.")
        return

    # Sequential path (default): keep original behavior
    if args.num_workers <= 1:
        device = args.device
        model, processor, blip_processor, vis_processors, text_processors, use_local_blip = _build_model_and_processors(args, device)

        with open(resolved.score_file_path, 'a', encoding='utf-8') as f_score, open(resolved.frame_file_path, 'a', encoding='utf-8') as f_frame:
            for idx in range(processed_count, total_videos):
                data = datas[idx]
                print(f"[{idx + 1}/{total_videos}] Processing video: {data.get('video_path', data.get('videoID', 'unknown'))}")
                score, frame_num, _ = _process_one_item(
                    args=args,
                    dataset_name=args.dataset_name,
                    video_root=resolved.video_root,
                    data=data,
                    device=device,
                    model=model,
                    processor=processor,
                    blip_processor=blip_processor,
                    vis_processors=vis_processors,
                    text_processors=text_processors,
                    use_local_blip=use_local_blip,
                )
                f_score.write(json.dumps(score) + "\n")
                f_frame.write(json.dumps(frame_num) + "\n")
                f_score.flush()
                f_frame.flush()

        print("Processing complete.")
        return

    # --- Parallel path (new) ---
    # Determine GPUs and worker mapping
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if args.device.startswith("cuda") and torch.cuda.is_available() and len(gpu_ids) == 0:
        # If CUDA is available but gpu_ids empty (e.g., user didn't specify), fall back to visible 0..N-1.
        gpu_ids = list(range(torch.cuda.device_count()))

    if args.device.startswith("cuda") and torch.cuda.is_available() and len(gpu_ids) > 0:
        effective_workers = max(1, len(gpu_ids) * max(1, args.workers_per_gpu))
    else:
        gpu_ids = []
        effective_workers = max(1, args.num_workers)

    # If user explicitly set num_workers > 1, allow it to cap workers (still supports multi-copy)
    effective_workers = min(effective_workers, max(1, args.num_workers))

    part_dir = args.part_dir.strip() if args.part_dir else os.path.join(resolved.out_score_path, "parts")
    os.makedirs(part_dir, exist_ok=True)

    # Make ranges starting from processed_count
    ranges = _make_ranges(processed_count, total_videos, args.chunk_size)
    if not ranges:
        print("Processing complete.")
        return

    print(f"Parallel mode: total={total_videos}, resume_from={processed_count}, chunks={len(ranges)}, workers={effective_workers}, gpus={gpu_ids or 'cpu'}")

    # Launch a fixed number of long-lived workers: each loads the model once and consumes tasks from a queue.
    ctx = mp.get_context("spawn")
    args_dict = vars(args).copy()

    expanded_gpus: List[Optional[int]] = []
    if gpu_ids:
        for gid in gpu_ids:
            for _ in range(max(1, args.workers_per_gpu)):
                expanded_gpus.append(gid)
    else:
        expanded_gpus = [None]

    task_queue = ctx.Queue()
    for r in ranges:
        task_queue.put(r)
    for _ in range(effective_workers):
        task_queue.put(None)  # sentinel

    workers = []
    for wid in range(effective_workers):
        gpu_id = expanded_gpus[wid % len(expanded_gpus)]
        p = ctx.Process(
            target=_worker_loop,
            kwargs=dict(
                worker_id=wid,
                gpu_id=gpu_id,
                args_dict=args_dict,
                resolved=resolved,
                part_dir=part_dir,
                task_queue=task_queue,
            ),
        )
        p.daemon = False
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A worker failed with exitcode {p.exitcode}")

    # Merge part files in order by start index, appending to final outputs
    with open(resolved.score_file_path, 'a', encoding='utf-8') as f_score, open(resolved.frame_file_path, 'a', encoding='utf-8') as f_frame:
        for start_idx, end_idx in ranges:
            score_part_path = os.path.join(part_dir, f"scores.part_{start_idx:09d}_{end_idx:09d}.jsonl")
            frame_part_path = os.path.join(part_dir, f"frames.part_{start_idx:09d}_{end_idx:09d}.jsonl")
            expected = max(0, end_idx - start_idx)
            if _count_existing_lines(score_part_path) != expected or _count_existing_lines(frame_part_path) != expected:
                raise RuntimeError(f"Part file line count mismatch for {start_idx}-{end_idx}. Please rerun to regenerate parts.")
            with open(score_part_path, 'r', encoding='utf-8') as ps, open(frame_part_path, 'r', encoding='utf-8') as pf:
                for sline, fline in zip(ps, pf):
                    f_score.write(sline.rstrip("\n") + "\n")
                    f_frame.write(fline.rstrip("\n") + "\n")
            f_score.flush()
            f_frame.flush()

    print("Processing complete.")

if __name__ == '__main__':
    mp.freeze_support()
    args = parse_arguments()
    main(args)