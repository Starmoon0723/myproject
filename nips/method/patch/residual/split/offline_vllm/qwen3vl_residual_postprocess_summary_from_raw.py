#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 summary only (offline-vLLM aligned):
- Read raw residual tensors produced by stage-1.
- Apply frame sampling + resize rules aligned with overwrite_vision_process.fetch_video().
- Do NOT save processed tensors, only dump per-sample summary JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qwen3vl_residual_tensor_builder_hf_aligned as core  # noqa: E402


MAX_RATIO = 200


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def offline_smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    if max_pixels < min_pixels:
        raise ValueError(f"max_pixels must be >= min_pixels, got {max_pixels} < {min_pixels}")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


@dataclass
class OfflineSummaryConfig:
    input_dir: str
    jsonl_path: str
    workers: int = min(max(os.cpu_count() or 4, 1), 4)

    patch_size: int = core.DEFAULT_PATCH_SIZE
    temporal_patch_size: int = core.DEFAULT_TEMPORAL_PATCH_SIZE
    spatial_merge_size: int = core.DEFAULT_SPATIAL_MERGE_SIZE

    frame_factor: int = 2
    sampling_fps: Optional[float] = core.DEFAULT_PROCESSOR_FPS
    sampling_num_frames: Optional[int] = None
    min_frames: int = 4
    max_frames: int = 2048
    min_pixels: int = 49152
    max_pixels: int = 655360
    total_pixels: int = 117440512
    do_sample_frames: bool = True
    video_frame_max_pixels: Optional[int] = None

    limit_tensors: int = 0


def find_raw_tensors(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.pt") if p.is_file()])


def _build_fallback_frame_times(total_frames: int, fps: float) -> List[float]:
    if fps <= 0:
        fps = core.DEFAULT_PROCESSOR_FPS
    return [float(i / fps) for i in range(total_frames)]


def _offline_sample_frame_indices(
    *,
    total_num_frames: int,
    metadata_fps: float,
    cfg: OfflineSummaryConfig,
) -> Tuple[List[int], Dict[str, Any]]:
    if total_num_frames <= 0:
        raise ValueError(f"total_num_frames must be positive, got {total_num_frames}")

    if not cfg.do_sample_frames:
        indices = list(range(total_num_frames))
        return indices, {
            "do_sample_frames": False,
            "sampling_source": "offline_keep_all",
            "frame_factor": int(cfg.frame_factor),
        }

    if cfg.sampling_num_frames is not None and cfg.sampling_fps is not None:
        raise ValueError("sampling_num_frames and sampling_fps are mutually exclusive.")

    frame_factor = int(cfg.frame_factor)
    if frame_factor <= 0:
        raise ValueError(f"frame_factor must be positive, got {frame_factor}")
    if total_num_frames < frame_factor:
        raise ValueError(
            f"total_num_frames={total_num_frames} is smaller than frame_factor={frame_factor}, cannot sample."
        )

    if cfg.sampling_num_frames is not None:
        nframes = round_by_factor(float(cfg.sampling_num_frames), frame_factor)
        nframes = int(min(max(nframes, frame_factor), total_num_frames))
        nframes = int(floor_by_factor(nframes, frame_factor))
        if nframes < frame_factor:
            nframes = frame_factor
        indices = np.linspace(0, total_num_frames - 1, nframes).round().astype(int).tolist()
        return indices, {
            "do_sample_frames": True,
            "sampling_source": "offline_nframes_uniform",
            "requested_sampling_num_frames": int(cfg.sampling_num_frames),
            "frame_factor": frame_factor,
        }

    video_fps = float(metadata_fps if metadata_fps and metadata_fps > 0 else core.DEFAULT_PROCESSOR_FPS)
    fps = float(cfg.sampling_fps if cfg.sampling_fps is not None else core.DEFAULT_PROCESSOR_FPS)
    min_frames = int(ceil_by_factor(cfg.min_frames, frame_factor))
    max_frames = int(floor_by_factor(min(cfg.max_frames, total_num_frames), frame_factor))
    if max_frames < frame_factor:
        max_frames = frame_factor

    nframes = total_num_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_num_frames)
    nframes = floor_by_factor(nframes, frame_factor)
    nframes = int(nframes)
    if not (frame_factor <= nframes <= total_num_frames):
        raise ValueError(f"nframes should be in [{frame_factor}, {total_num_frames}], but got {nframes}.")

    indices = np.linspace(0, total_num_frames - 1, nframes).round().astype(int).tolist()
    return indices, {
        "do_sample_frames": True,
        "sampling_source": "offline_fps_uniform",
        "requested_sampling_fps": fps,
        "frame_factor": frame_factor,
        "min_frames_effective": min_frames,
        "max_frames_effective": max_frames,
        "video_fps_used": video_fps,
    }


def _offline_compute_resize(
    *,
    orig_h: int,
    orig_w: int,
    sampled_nframes: int,
    cfg: OfflineSummaryConfig,
) -> Tuple[int, int, Dict[str, Any]]:
    factor = int(cfg.patch_size * cfg.spatial_merge_size)
    video_frame_max_pixels = (
        int(cfg.video_frame_max_pixels)
        if cfg.video_frame_max_pixels is not None
        else int(768 * factor * factor)
    )
    dynamic_cap = max(
        min(video_frame_max_pixels, int(cfg.total_pixels / max(sampled_nframes, 1) * cfg.frame_factor)),
        int(cfg.min_pixels * 1.05),
    )
    effective_max_pixels = min(int(cfg.max_pixels), int(dynamic_cap))

    resized_h, resized_w = offline_smart_resize(
        orig_h,
        orig_w,
        factor=factor,
        min_pixels=int(cfg.min_pixels),
        max_pixels=int(effective_max_pixels),
    )
    meta = {
        "resize_factor": factor,
        "video_frame_max_pixels": int(video_frame_max_pixels),
        "min_pixels": int(cfg.min_pixels),
        "max_pixels_requested": int(cfg.max_pixels),
        "total_pixels": int(cfg.total_pixels),
        "dynamic_max_pixels_cap": int(dynamic_cap),
        "effective_max_pixels": int(effective_max_pixels),
    }
    return resized_h, resized_w, meta


def summarize_one_raw_tensor(raw_path_str: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = OfflineSummaryConfig(**cfg_dict)
    raw_path = Path(raw_path_str)

    payload = torch.load(raw_path, map_location="cpu")
    meta = payload.get("metadata", {}) or {}

    raw_residual = payload.get("raw_residual_gray_uint8", None)
    total_frames = int(meta.get("original_total_frames", 0))
    orig_h = int(meta.get("original_height", 0))
    orig_w = int(meta.get("original_width", 0))

    # fallback to tensor shape when metadata is missing
    if (
        isinstance(raw_residual, torch.Tensor)
        and raw_residual.ndim == 3
        and (total_frames <= 0 or orig_h <= 0 or orig_w <= 0)
    ):
        total_frames = int(raw_residual.shape[0])
        orig_h = int(raw_residual.shape[1])
        orig_w = int(raw_residual.shape[2])

    if total_frames <= 0 or orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid metadata in {raw_path}: total_frames={total_frames}, H={orig_h}, W={orig_w}")

    decoded_video_fps = float(meta.get("decoded_video_fps", core.DEFAULT_PROCESSOR_FPS))
    frame_times_sec = list(meta.get("frame_times_sec", []))
    frame_pict_types = list(meta.get("frame_pict_types", []))
    if len(frame_times_sec) != total_frames:
        frame_times_sec = _build_fallback_frame_times(total_frames, decoded_video_fps)
    if len(frame_pict_types) != total_frames:
        frame_pict_types = ["?"] * total_frames

    sample_idx, sampling_meta = _offline_sample_frame_indices(
        total_num_frames=total_frames,
        metadata_fps=decoded_video_fps,
        cfg=cfg,
    )
    sampled_nframes = int(len(sample_idx))
    sampled_times = [frame_times_sec[i] for i in sample_idx]
    sampled_ptypes = [frame_pict_types[i] for i in sample_idx]

    resized_h, resized_w, resize_meta = _offline_compute_resize(
        orig_h=orig_h,
        orig_w=orig_w,
        sampled_nframes=sampled_nframes,
        cfg=cfg,
    )

    vision_grid_h = int(resized_h // cfg.patch_size)
    vision_grid_w = int(resized_w // cfg.patch_size)
    llm_cell = int(cfg.patch_size * cfg.spatial_merge_size)
    llm_grid_h = int(resized_h // llm_cell)
    llm_grid_w = int(resized_w // llm_cell)

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

    vision_tokens_per_temporal_block = int(vision_grid_h * vision_grid_w)
    total_vision_tokens = int(grid_t * vision_tokens_per_temporal_block)
    llm_tokens_per_temporal_block = int(llm_grid_h * llm_grid_w)
    total_llm_tokens = int(grid_t * llm_tokens_per_temporal_block)

    return {
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
            "min_frames": int(cfg.min_frames),
            "max_frames": int(cfg.max_frames),
            "frame_factor": int(cfg.frame_factor),
            **sampling_meta,
        },
        "resize_alignment": {
            "min_pixels": int(cfg.min_pixels),
            "max_pixels": int(cfg.max_pixels),
            "total_pixels": int(cfg.total_pixels),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-2 summary only: offline-vLLM aligned sampling/resize from raw residual.",
    )
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--jsonl_path", type=Path, required=True)
    p.add_argument("--workers", type=int, default=min(max(os.cpu_count() or 4, 1), 4))

    p.add_argument("--patch_size", type=int, default=core.DEFAULT_PATCH_SIZE)
    p.add_argument("--temporal_patch_size", type=int, default=core.DEFAULT_TEMPORAL_PATCH_SIZE)
    p.add_argument("--spatial_merge_size", type=int, default=core.DEFAULT_SPATIAL_MERGE_SIZE)
    p.add_argument("--frame_factor", type=int, default=2)

    p.add_argument("--do_sample_frames", action="store_true", default=True)
    p.add_argument("--no_do_sample_frames", action="store_true")
    p.add_argument("--sampling_fps", type=float, default=core.DEFAULT_PROCESSOR_FPS)
    p.add_argument("--sampling_num_frames", type=int, default=None)
    p.add_argument("--min_frames", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=2048)
    p.add_argument("--min_pixels", type=int, default=49152)
    p.add_argument("--max_pixels", type=int, default=655360)
    p.add_argument("--total_pixels", type=int, default=117440512)
    p.add_argument("--video_frame_max_pixels", type=int, default=None)

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
    cfg = OfflineSummaryConfig(
        input_dir=str(input_dir),
        jsonl_path=str(jsonl_path),
        workers=max(1, int(args.workers)),
        patch_size=int(args.patch_size),
        temporal_patch_size=int(args.temporal_patch_size),
        spatial_merge_size=int(args.spatial_merge_size),
        frame_factor=int(args.frame_factor),
        do_sample_frames=bool(do_sample_frames),
        sampling_fps=None if sampling_fps is None else float(sampling_fps),
        sampling_num_frames=None if args.sampling_num_frames is None else int(args.sampling_num_frames),
        min_frames=int(args.min_frames),
        max_frames=int(args.max_frames),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        total_pixels=int(args.total_pixels),
        video_frame_max_pixels=None if args.video_frame_max_pixels is None else int(args.video_frame_max_pixels),
        limit_tensors=int(args.limit_tensors),
    )

    print(f"[INFO] Found {len(tensors)} raw tensors under {input_dir}")
    print(f"[INFO] JSONL output: {jsonl_path}")
    print(
        "[INFO] fps=%s nframes=%s min/max_frames=(%d,%d) min/max/total_pixels=(%d,%d,%d)"
        % (
            str(cfg.sampling_fps),
            str(cfg.sampling_num_frames),
            cfg.min_frames,
            cfg.max_frames,
            cfg.min_pixels,
            cfg.max_pixels,
            cfg.total_pixels,
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
                summaries.append({"source_raw_path": str(source), "error": str(exc)})
                print(f"[ERR] {source}: {exc}", file=sys.stderr)

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
