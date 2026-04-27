#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 summary (no tensor saving):
- 读取第一步生成的 raw residual .pt。
- 使用与 HF 对齐的抽帧 + smart resize + 时间分组逻辑。
- 不生成/保存任何新张量，只把关键信息写入一个 JSONL 文件，便于和 vLLM 对齐核对。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qwen3vl_residual_tensor_builder_hf_aligned as core  # noqa: E402


@dataclass
class SummaryConfig:
    input_dir: str
    jsonl_path: str
    workers: int = min(max(os.cpu_count() or 4, 1), 4)

    # 以下参数需与实际推理/processor 对齐
    patch_size: int = core.DEFAULT_PATCH_SIZE
    temporal_patch_size: int = core.DEFAULT_TEMPORAL_PATCH_SIZE
    spatial_merge_size: int = core.DEFAULT_SPATIAL_MERGE_SIZE
    processor_shortest_edge: int = core.DEFAULT_VIDEO_SHORTEST_EDGE
    processor_longest_edge: int = core.DEFAULT_VIDEO_LONGEST_EDGE
    processor_fps: float = core.DEFAULT_PROCESSOR_FPS
    processor_min_frames: int = core.DEFAULT_PROCESSOR_MIN_FRAMES
    processor_max_frames: int = core.DEFAULT_PROCESSOR_MAX_FRAMES

    do_sample_frames: bool = True
    sampling_fps: Optional[float] = core.DEFAULT_PROCESSOR_FPS
    sampling_num_frames: Optional[int] = None
    runtime_max_frames: Optional[int] = core.DEFAULT_RUNTIME_MAX_FRAMES
    runtime_max_frames_policy: Literal["ignore", "override", "min"] = "override"

    limit_tensors: int = 0


def _to_core_config(cfg: SummaryConfig) -> core.Config:
    """把本地 SummaryConfig 映射为原脚本的 Config，用于调用 HF 对齐函数。"""
    return core.Config(
        input_dir=cfg.input_dir,
        output_dir=cfg.input_dir,
        workers=cfg.workers,
        patch_size=cfg.patch_size,
        temporal_patch_size=cfg.temporal_patch_size,
        spatial_merge_size=cfg.spatial_merge_size,
        processor_shortest_edge=cfg.processor_shortest_edge,
        processor_longest_edge=cfg.processor_longest_edge,
        processor_fps=cfg.processor_fps,
        processor_min_frames=cfg.processor_min_frames,
        processor_max_frames=cfg.processor_max_frames,
        do_sample_frames=cfg.do_sample_frames,
        sampling_fps=cfg.sampling_fps,
        sampling_num_frames=cfg.sampling_num_frames,
        runtime_max_frames=cfg.runtime_max_frames,
        runtime_max_frames_policy=cfg.runtime_max_frames_policy,
        save_video_uint8=False,
        save_residual_gray_uint8=False,
        save_vision_residual_grid_uint8=True,
        save_llm_residual_grid_uint8=True,
        preview_frames=0,
        residual_resize_mode="bicubic",
        video_resize_mode="bicubic",
    )


def find_raw_tensors(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.pt") if p.is_file()])


def _build_fallback_frame_times(total_frames: int, fps: float) -> List[float]:
    if fps <= 0:
        fps = core.DEFAULT_PROCESSOR_FPS
    return [float(i / fps) for i in range(total_frames)]


def summarize_one_raw_tensor(raw_path_str: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = SummaryConfig(**cfg_dict)
    core_cfg = _to_core_config(cfg)

    raw_path = Path(raw_path_str)

    payload = torch.load(raw_path, map_location="cpu")
    meta = payload.get("metadata", {}) or {}

    total_frames = int(meta.get("original_total_frames", 0))
    orig_h = int(meta.get("original_height", 0))
    orig_w = int(meta.get("original_width", 0))
    if total_frames <= 0 or orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid metadata in {raw_path}: total_frames={total_frames}, H={orig_h}, W={orig_w}")

    decoded_video_fps = float(meta.get("decoded_video_fps", core.DEFAULT_PROCESSOR_FPS))
    frame_times_sec = list(meta.get("frame_times_sec", []))
    frame_pict_types = list(meta.get("frame_pict_types", []))

    if len(frame_times_sec) != total_frames:
        frame_times_sec = _build_fallback_frame_times(total_frames, decoded_video_fps)
    if len(frame_pict_types) != total_frames:
        frame_pict_types = ["?"] * total_frames

    # 1) HF 抽帧
    sample_idx, sampling_meta = core.hf_sample_frame_indices(
        total_num_frames=total_frames,
        metadata_fps=decoded_video_fps,
        cfg=core_cfg,
    )
    sampled_nframes = int(len(sample_idx))
    sampled_times = [frame_times_sec[i] for i in sample_idx]
    sampled_ptypes = [frame_pict_types[i] for i in sample_idx]

    # 2) smart resize 计算输出高宽（不真正 resize）
    resized_h, resized_w, resize_meta = core.compute_hf_video_resize(orig_h, orig_w, sampled_nframes, core_cfg)

    vision_grid_h = int(resized_h // cfg.patch_size)
    vision_grid_w = int(resized_w // cfg.patch_size)
    llm_cell = int(cfg.patch_size * cfg.spatial_merge_size)
    llm_grid_h = int(resized_h // llm_cell)
    llm_grid_w = int(resized_w // llm_cell)

    # 3) 时间维 padding + 分组（与原脚本一致）
    temporal_pad = int((-sampled_nframes) % cfg.temporal_patch_size)
    padded_nframes = int(sampled_nframes + temporal_pad)
    grid_t = max(padded_nframes // cfg.temporal_patch_size, 1)
    padded_sample_idx = sample_idx + ([sample_idx[-1]] * temporal_pad if temporal_pad > 0 else [])
    padded_sampled_times = sampled_times + ([sampled_times[-1]] * temporal_pad if temporal_pad > 0 else [])

    temporal_groups: List[List[int]] = []
    temporal_group_times_sec: List[float] = []
    for i in range(0, padded_nframes, cfg.temporal_patch_size):
        frame_group = padded_sample_idx[i : i + cfg.temporal_patch_size]
        time_group = padded_sampled_times[i : i + cfg.temporal_patch_size]
        temporal_groups.append(frame_group)
        temporal_group_times_sec.append(float((time_group[0] + time_group[-1]) / 2.0))

    # 4) token 统计（便于和 vLLM 对比）
    # 视觉 patch 级网格：每个时间块有 vision_grid_h * vision_grid_w 个 patch
    vision_tokens_per_temporal_block = int(vision_grid_h * vision_grid_w)
    total_vision_tokens = int(grid_t * vision_tokens_per_temporal_block)

    # LLM 级：按 spatial_merge_size 合并空间 patch
    llm_tokens_per_temporal_block = int(llm_grid_h * llm_grid_w)
    total_llm_tokens = int(grid_t * llm_tokens_per_temporal_block)

    summary: Dict[str, Any] = {
        "source_raw_path": str(raw_path),
        "source_video_path": str(meta.get("video_path", "")),
        "source_video_rel_path": str(meta.get("video_rel_path", "")),
        "original_total_frames": int(total_frames),
        "decoded_video_fps": float(decoded_video_fps),
        "original_height": int(orig_h),
        "original_width": int(orig_w),
        "sampled_indices": sample_idx,
        "sampled_nframes_before_temporal_pad": int(sampled_nframes),
        "sampled_times_sec": sampled_times,
        "sampled_pict_types": sampled_ptypes,
        "temporal_patch_size": int(cfg.temporal_patch_size),
        "temporal_pad_frames": int(temporal_pad),
        "temporal_padded_nframes": int(padded_nframes),
        "temporal_group_count": int(grid_t),
        "temporal_group_frame_indices": temporal_groups,
        "temporal_group_mid_times_sec": temporal_group_times_sec,
        "resized_height": int(resized_h),
        "resized_width": int(resized_w),
        "patch_size": int(cfg.patch_size),
        "spatial_merge_size": int(cfg.spatial_merge_size),
        "vision_patch_grid_hw": [vision_grid_h, vision_grid_w],
        "llm_grid_hw": [llm_grid_h, llm_grid_w],
        "video_grid_thw": [grid_t, vision_grid_h, vision_grid_w],
        "vision_tokens_per_temporal_block": vision_tokens_per_temporal_block,
        "total_vision_tokens": total_vision_tokens,
        "llm_tokens_per_temporal_block": llm_tokens_per_temporal_block,
        "total_llm_tokens": total_llm_tokens,
        "sampling_alignment": {
            "do_sample_frames": bool(cfg.do_sample_frames),
            "sampling_fps": cfg.sampling_fps,
            "sampling_num_frames": cfg.sampling_num_frames,
            "processor_fps": cfg.processor_fps,
            "processor_min_frames": int(cfg.processor_min_frames),
            "processor_max_frames": int(cfg.processor_max_frames),
            "runtime_max_frames": cfg.runtime_max_frames,
            "runtime_max_frames_policy": cfg.runtime_max_frames_policy,
            **sampling_meta,
        },
        "resize_alignment": {
            "processor_shortest_edge": int(cfg.processor_shortest_edge),
            "processor_longest_edge": int(cfg.processor_longest_edge),
            **resize_meta,
        },
        "residual_definition": meta.get(
            "residual_definition",
            "abs(curr_luma - warp(prev_luma, mv_blocks_of_curr))",
        ),
        "mv_backend": meta.get("mv_backend", "pyav_ffmpeg_side_data"),
        "num_i_frames": int(meta.get("num_i_frames", 0)),
        "num_p_frames": int(meta.get("num_p_frames", 0)),
        "num_other_frames": int(meta.get("num_other_frames", 0)),
        "num_frames_with_mv": int(meta.get("num_frames_with_mv", 0)),
        "num_frames_without_mv": int(meta.get("num_frames_without_mv", 0)),
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-2 (summary only): 对 raw residual 做 HF 抽帧+resize 逻辑推演，仅输出 JSONL 统计信息。",
    )
    p.add_argument("--input_dir", type=Path, required=True, help="第一步 raw 残差 .pt 所在目录")
    p.add_argument("--jsonl_path", type=Path, required=True, help="输出 JSONL 文件路径")
    p.add_argument("--workers", type=int, default=min(max(os.cpu_count() or 4, 1), 4))

    p.add_argument("--patch_size", type=int, default=core.DEFAULT_PATCH_SIZE)
    p.add_argument("--temporal_patch_size", type=int, default=core.DEFAULT_TEMPORAL_PATCH_SIZE)
    p.add_argument("--spatial_merge_size", type=int, default=core.DEFAULT_SPATIAL_MERGE_SIZE)
    p.add_argument("--processor_shortest_edge", type=int, default=core.DEFAULT_VIDEO_SHORTEST_EDGE)
    p.add_argument("--processor_longest_edge", type=int, default=core.DEFAULT_VIDEO_LONGEST_EDGE)
    p.add_argument("--processor_fps", type=float, default=core.DEFAULT_PROCESSOR_FPS)
    p.add_argument("--processor_min_frames", type=int, default=core.DEFAULT_PROCESSOR_MIN_FRAMES)
    p.add_argument("--processor_max_frames", type=int, default=core.DEFAULT_PROCESSOR_MAX_FRAMES)

    p.add_argument("--do_sample_frames", action="store_true", default=True)
    p.add_argument("--no_do_sample_frames", action="store_true")
    p.add_argument("--sampling_fps", type=float, default=core.DEFAULT_PROCESSOR_FPS)
    p.add_argument("--sampling_num_frames", type=int, default=None)
    p.add_argument("--runtime_max_frames", type=int, default=core.DEFAULT_RUNTIME_MAX_FRAMES)
    p.add_argument(
        "--runtime_max_frames_policy",
        type=str,
        default="override",
        choices=["ignore", "override", "min"],
    )

    p.add_argument("--limit_tensors", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    jsonl_path = args.jsonl_path

    tensors = find_raw_tensors(input_dir)
    if args.limit_tensors and args.limit_tensors > 0:
        tensors = tensors[: args.limit_tensors]
    if not tensors:
        print(f"No .pt files found under: {input_dir}", file=sys.stderr)
        return 1

    sampling_fps = None if args.sampling_num_frames is not None else args.sampling_fps
    do_sample_frames = False if args.no_do_sample_frames else bool(args.do_sample_frames)

    cfg = SummaryConfig(
        input_dir=str(input_dir),
        jsonl_path=str(jsonl_path),
        workers=max(1, int(args.workers)),
        patch_size=int(args.patch_size),
        temporal_patch_size=int(args.temporal_patch_size),
        spatial_merge_size=int(args.spatial_merge_size),
        processor_shortest_edge=int(args.processor_shortest_edge),
        processor_longest_edge=int(args.processor_longest_edge),
        processor_fps=float(args.processor_fps),
        processor_min_frames=int(args.processor_min_frames),
        processor_max_frames=int(args.processor_max_frames),
        do_sample_frames=bool(do_sample_frames),
        sampling_fps=None if sampling_fps is None else float(sampling_fps),
        sampling_num_frames=None if args.sampling_num_frames is None else int(args.sampling_num_frames),
        runtime_max_frames=None if args.runtime_max_frames is None else int(args.runtime_max_frames),
        runtime_max_frames_policy=str(args.runtime_max_frames_policy),
        limit_tensors=int(args.limit_tensors),
    )

    print(f"[INFO] Found {len(tensors)} raw tensors under {input_dir}")
    print(f"[INFO] JSONL output: {jsonl_path}")
    print(
        "[INFO] patch_size=%d, temporal_patch_size=%d, spatial_merge_size=%d, "
        "sampling_fps=%s, processor_max_frames=%d, runtime_max_frames=%s, policy=%s"
        % (
            cfg.patch_size,
            cfg.temporal_patch_size,
            cfg.spatial_merge_size,
            str(cfg.sampling_fps),
            cfg.processor_max_frames,
            str(cfg.runtime_max_frames),
            cfg.runtime_max_frames_policy,
        )
    )

    summaries: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(summarize_one_raw_tensor, str(p), asdict(cfg)): p for p in tensors}
        for fut in as_completed(futures):
            source = futures[fut]
            try:
                summary = fut.result()
                summaries.append(summary)
                print(f"[OK] {source}")
            except Exception as exc:
                err_info = {
                    "source_raw_path": str(source),
                    "error": str(exc),
                }
                summaries.append(err_info)
                print(f"[ERR] {source}: {exc}", file=sys.stderr)

    # 写 JSONL
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in summaries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    ok_count = sum(1 for s in summaries if "error" not in s)
    print(f"[DONE] summarized={ok_count} / {len(summaries)}")
    print(f"[DONE] jsonl: {jsonl_path}")
    return 0 if ok_count > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

