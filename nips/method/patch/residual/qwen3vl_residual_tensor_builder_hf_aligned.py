#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Qwen3-VL-aligned per-video tensors and approximate motion-compensated residual maps.

This revision keeps the original decode / motion-vector residual / resize / save workflow,
but replaces the frame-sampling and resize logic with a strict reproduction of the current
Hugging Face Qwen3VLVideoProcessor video path, while also exposing knobs needed to mimic
common vLLM runtime overrides.

What is aligned here
--------------------
1) I/O-stage no-sampling assumption:
   the input video is fully decoded first (equivalent to vLLM media_io_kwargs video.num_frames=-1).
2) HF-stage frame sampling:
   reproduces Qwen3VLVideoProcessor.sample_frames() behavior for fps / num_frames uniform sampling.
3) HF-stage smart resize:
   reproduces transformers.models.qwen3_vl.video_processing_qwen3_vl.smart_resize().
4) Temporal padding metadata:
   reproduces the temporal_patch_size padding rule used before video_grid_thw is formed.

What is intentionally unchanged
-------------------------------
- Residual definition stays:
    abs(curr_luma - warp(prev_luma, mv_blocks_of_curr))
- Motion-vector extraction stays PyAV + FFmpeg side-data based.
- Compact storage stays token-grid oriented by default.

Outputs per video (.pt)
-----------------------
Default:
- vision_residual_grid_uint8: ByteTensor [T, H/patch, W/patch]
- llm_residual_grid_uint8:    ByteTensor [T, H/(patch*merge), W/(patch*merge)]
- metadata:                  dict

Optional heavy/debug outputs:
- residual_gray_uint8:       ByteTensor [T, H, W]
- video_uint8:               ByteTensor [T, 3, H, W]
- preview_video_uint8:       ByteTensor [P, 3, H, W]
- preview_residual_gray_uint8: ByteTensor [P, H, W]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import av  # type: ignore
except Exception:
    av = None


# -----------------------------
# Defaults mirrored from current Qwen3-VL processor / model files
# -----------------------------

MAX_RATIO = 200
DEFAULT_PATCH_SIZE = 16
DEFAULT_TEMPORAL_PATCH_SIZE = 2
DEFAULT_SPATIAL_MERGE_SIZE = 2
DEFAULT_PROCESSOR_FPS = 2.0
DEFAULT_PROCESSOR_MIN_FRAMES = 4
DEFAULT_PROCESSOR_MAX_FRAMES = 2048
DEFAULT_VIDEO_SHORTEST_EDGE = 4096
DEFAULT_VIDEO_LONGEST_EDGE = 25165824   # 469762048
DEFAULT_RUNTIME_MAX_FRAMES = 2048


@dataclass
class Config:
    input_dir: str
    output_dir: str
    workers: int = max(os.cpu_count() or 4, 4)

    # Vision / processor parameters
    patch_size: int = DEFAULT_PATCH_SIZE
    temporal_patch_size: int = DEFAULT_TEMPORAL_PATCH_SIZE
    spatial_merge_size: int = DEFAULT_SPATIAL_MERGE_SIZE
    processor_shortest_edge: int = DEFAULT_VIDEO_SHORTEST_EDGE
    processor_longest_edge: int = DEFAULT_VIDEO_LONGEST_EDGE
    processor_fps: float = DEFAULT_PROCESSOR_FPS
    processor_min_frames: int = DEFAULT_PROCESSOR_MIN_FRAMES
    processor_max_frames: int = DEFAULT_PROCESSOR_MAX_FRAMES

    # Request-time behavior to mimic online vLLM + HF processor usage
    do_sample_frames: bool = True
    sampling_fps: Optional[float] = DEFAULT_PROCESSOR_FPS
    sampling_num_frames: Optional[int] = None
    runtime_max_frames: Optional[int] = DEFAULT_RUNTIME_MAX_FRAMES
    runtime_max_frames_policy: Literal["ignore", "override", "min"] = "override"

    # Save controls
    save_video_uint8: bool = False
    save_residual_gray_uint8: bool = False
    save_vision_residual_grid_uint8: bool = True
    save_llm_residual_grid_uint8: bool = True
    preview_frames: int = 0
    residual_resize_mode: str = "bicubic"
    video_resize_mode: str = "bicubic"


@dataclass
class DecodedFrame:
    index: int
    time_sec: float
    pict_type: str
    rgb: np.ndarray  # [H, W, 3], uint8
    mv_blocks: List[Tuple[int, int, int, int, int, int]]  # src_x, src_y, dst_x, dst_y, w, h


# -----------------------------
# HF-aligned helpers
# -----------------------------


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor



def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor



def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor



def hf_video_smart_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """Reproduce transformers.models.qwen3_vl.video_processing_qwen3_vl.smart_resize."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    t_bar = ceil_by_factor(num_frames, temporal_factor)

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return int(h_bar), int(w_bar)



def compute_effective_max_frames(cfg: Config) -> int:
    runtime_max = cfg.runtime_max_frames
    processor_max = cfg.processor_max_frames

    if runtime_max is None or cfg.runtime_max_frames_policy == "ignore":
        return int(processor_max)
    if cfg.runtime_max_frames_policy == "override":
        return int(runtime_max)
    if cfg.runtime_max_frames_policy == "min":
        return int(min(processor_max, runtime_max))
    raise ValueError(f"Unsupported runtime_max_frames_policy={cfg.runtime_max_frames_policy}")



def hf_sample_frame_indices(
    *,
    total_num_frames: int,
    metadata_fps: Optional[float],
    cfg: Config,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Reproduce the effective HF-stage frame sampling.

    Important:
    - If do_sample_frames is False, we keep all decoded frames from the I/O layer.
    - If do_sample_frames is True:
        * sampling_num_frames and sampling_fps are mutually exclusive.
        * when sampling_fps is used, num_frames = int(total_num_frames / metadata_fps * sampling_fps)
        * clamp follows HF sample_frames, with an explicit runtime max cap policy exposed here.
    """
    if total_num_frames <= 0:
        raise ValueError(f"total_num_frames must be positive, got {total_num_frames}")

    if not cfg.do_sample_frames:
        indices = list(range(total_num_frames))
        return indices, {
            "do_sample_frames": False,
            "sampling_source": "io_layer_all_frames",
            "effective_max_frames": None,
        }

    if cfg.sampling_num_frames is not None and cfg.sampling_fps is not None:
        raise ValueError("sampling_num_frames and sampling_fps are mutually exclusive, matching HF behavior.")

    num_frames: Optional[int] = cfg.sampling_num_frames
    used_metadata_fps = metadata_fps
    if num_frames is None and cfg.sampling_fps is not None:
        if used_metadata_fps is None or used_metadata_fps <= 0:
            used_metadata_fps = 24.0
        num_frames = int(total_num_frames / used_metadata_fps * float(cfg.sampling_fps))

    effective_max_frames = compute_effective_max_frames(cfg)

    if num_frames is None:
        num_frames = min(max(total_num_frames, cfg.processor_min_frames), effective_max_frames)
    else:
        num_frames = min(max(num_frames, cfg.processor_min_frames), effective_max_frames, total_num_frames)

    # HF uses np.linspace(...).round().astype(int)
    indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int).tolist()
    return indices, {
        "do_sample_frames": True,
        "sampling_source": "hf_processor_uniform",
        "requested_sampling_fps": cfg.sampling_fps,
        "requested_sampling_num_frames": cfg.sampling_num_frames,
        "metadata_fps_used_for_sampling": used_metadata_fps,
        "effective_max_frames": effective_max_frames,
    }



def compute_hf_video_resize(height: int, width: int, sampled_num_frames: int, cfg: Config) -> Tuple[int, int, Dict[str, Any]]:
    resized_h, resized_w = hf_video_smart_resize(
        num_frames=sampled_num_frames,
        height=height,
        width=width,
        temporal_factor=cfg.temporal_patch_size,
        factor=cfg.patch_size * cfg.spatial_merge_size,
        min_pixels=cfg.processor_shortest_edge,
        max_pixels=cfg.processor_longest_edge,
    )
    meta = {
        "processor_shortest_edge": int(cfg.processor_shortest_edge),
        "processor_longest_edge": int(cfg.processor_longest_edge),
        "resize_factor": int(cfg.patch_size * cfg.spatial_merge_size),
    }
    return resized_h, resized_w, meta


# -----------------------------
# Video discovery
# -----------------------------


def find_videos(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.mp4") if p.is_file()])


# -----------------------------
# Motion-vector extraction utils
# -----------------------------


def _require_pyav() -> Any:
    if av is None:
        raise RuntimeError(
            "PyAV is required for MV-based residual extraction but is not installed. "
            "Install it first, e.g. `pip install av`."
        )
    return av



def _frame_time_seconds(frame: Any, stream: Any) -> float:
    t = getattr(frame, "time", None)
    if t is not None:
        try:
            return float(t)
        except Exception:
            pass
    pts = getattr(frame, "pts", None)
    time_base = getattr(stream, "time_base", None)
    if pts is not None and time_base is not None:
        try:
            return float(pts * time_base)
        except Exception:
            pass
    return 0.0



def _frame_pict_type(frame: Any) -> str:
    pict_type = getattr(frame, "pict_type", None)
    name = getattr(pict_type, "name", None)
    if name is not None:
        return str(name)
    if pict_type is None:
        return "?"
    return str(pict_type)



def _get_mv_side_data(frame: Any) -> Optional[Any]:
    try:
        for sd in frame.side_data:
            t = getattr(sd, "type", None)
            tname = getattr(t, "name", str(t))
            if "MOTION_VECTORS" in tname:
                return sd
    except Exception:
        pass
    try:
        return frame.side_data.get("MOTION_VECTORS")
    except Exception:
        return None



def _decode_video_with_pyav(video_path: Path) -> Tuple[List[DecodedFrame], float]:
    av_mod = _require_pyav()
    decoded: List[DecodedFrame] = []

    with av_mod.open(str(video_path)) as container:
        stream = container.streams.video[0]
        codec_ctx = stream.codec_context

        configured = False
        try:
            codec_ctx.flags2 |= av_mod.codec.context.Flags2.EXPORT_MVS
            configured = True
        except Exception:
            pass
        if not configured:
            try:
                opts = dict(getattr(codec_ctx, "options", {}) or {})
                opts["flags2"] = "+export_mvs"
                codec_ctx.options = opts
            except Exception:
                pass
        try:
            is_open = bool(getattr(codec_ctx, "is_open", False))
            if not is_open:
                codec_ctx.open()
        except Exception:
            pass

        avg_rate = getattr(stream, "average_rate", None)
        video_fps = float(avg_rate) if avg_rate is not None else 0.0

        frame_idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                rgb = frame.to_ndarray(format="rgb24")
                pict_type = _frame_pict_type(frame)
                time_sec = _frame_time_seconds(frame, stream)
                mv_side_data = _get_mv_side_data(frame)
                mv_blocks: List[Tuple[int, int, int, int, int, int]] = []
                if mv_side_data is not None:
                    for mv in mv_side_data:
                        src_x = int(getattr(mv, "src_x", 0))
                        src_y = int(getattr(mv, "src_y", 0))
                        dst_x = int(getattr(mv, "dst_x", 0))
                        dst_y = int(getattr(mv, "dst_y", 0))
                        w = int(getattr(mv, "w", 0))
                        h = int(getattr(mv, "h", 0))
                        if w <= 0 or h <= 0:
                            continue
                        mv_blocks.append((src_x, src_y, dst_x, dst_y, w, h))
                decoded.append(
                    DecodedFrame(
                        index=frame_idx,
                        time_sec=float(time_sec),
                        pict_type=pict_type,
                        rgb=np.ascontiguousarray(rgb),
                        mv_blocks=mv_blocks,
                    )
                )
                frame_idx += 1

    if video_fps <= 0 and len(decoded) >= 2:
        duration = max(decoded[-1].time_sec - decoded[0].time_sec, 1e-6)
        video_fps = (len(decoded) - 1) / duration
    if video_fps <= 0:
        video_fps = DEFAULT_PROCESSOR_FPS
    return decoded, float(video_fps)


# -----------------------------
# Residual building
# -----------------------------


def _rgb_to_luma_uint8(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)



def _copy_mv_block(pred: np.ndarray, prev: np.ndarray, sx: int, sy: int, dx: int, dy: int, bw: int, bh: int) -> None:
    h, w = prev.shape
    if bw <= 0 or bh <= 0:
        return

    if sx < 0:
        shift = -sx
        sx += shift
        dx += shift
        bw -= shift
    if sy < 0:
        shift = -sy
        sy += shift
        dy += shift
        bh -= shift
    if dx < 0:
        shift = -dx
        sx += shift
        dx += shift
        bw -= shift
    if dy < 0:
        shift = -dy
        sy += shift
        dy += shift
        bh -= shift

    if bw <= 0 or bh <= 0:
        return

    bw = min(bw, w - sx, w - dx)
    bh = min(bh, h - sy, h - dy)
    if bw <= 0 or bh <= 0:
        return

    pred[dy : dy + bh, dx : dx + bw] = prev[sy : sy + bh, sx : sx + bw]



def _warp_prev_luma_with_mvs(prev_luma: np.ndarray, mv_blocks: Sequence[Tuple[int, int, int, int, int, int]]) -> np.ndarray:
    pred = prev_luma.copy()
    for sx, sy, dx, dy, bw, bh in mv_blocks:
        _copy_mv_block(pred, prev_luma, sx, sy, dx, dy, bw, bh)
    return pred



def _build_original_residuals(decoded_frames: Sequence[DecodedFrame]) -> Tuple[np.ndarray, np.ndarray, List[str], List[float], Dict[str, int]]:
    if not decoded_frames:
        raise ValueError("No decoded frames found.")

    rgbs = [fr.rgb for fr in decoded_frames]
    pict_types = [fr.pict_type for fr in decoded_frames]
    times_sec = [float(fr.time_sec) for fr in decoded_frames]
    lumas = [_rgb_to_luma_uint8(fr.rgb) for fr in decoded_frames]

    h, w = lumas[0].shape
    residuals: List[np.ndarray] = []
    stats = {
        "num_i_frames": 0,
        "num_p_frames": 0,
        "num_other_frames": 0,
        "num_frames_with_mv": 0,
        "num_frames_without_mv": 0,
    }

    for i, fr in enumerate(decoded_frames):
        ptype = fr.pict_type.upper() if fr.pict_type is not None else "?"
        if ptype == "I":
            stats["num_i_frames"] += 1
        elif ptype == "P":
            stats["num_p_frames"] += 1
        else:
            stats["num_other_frames"] += 1

        if i == 0 or ptype == "I":
            residuals.append(np.zeros((h, w), dtype=np.uint8))
            continue

        prev_luma = lumas[i - 1]
        curr_luma = lumas[i]

        if fr.mv_blocks:
            stats["num_frames_with_mv"] += 1
            pred = _warp_prev_luma_with_mvs(prev_luma, fr.mv_blocks)
        else:
            stats["num_frames_without_mv"] += 1
            pred = prev_luma

        diff = np.abs(curr_luma.astype(np.int16) - pred.astype(np.int16)).astype(np.uint8)
        residuals.append(diff)

    rgb_video = np.stack(rgbs, axis=0)
    residual_gray = np.stack(residuals, axis=0)
    return rgb_video, residual_gray, pict_types, times_sec, stats


# -----------------------------
# Resize / pooling / save
# -----------------------------


def _interpolate_nchw(tensor: torch.Tensor, out_h: int, out_w: int, mode: str) -> torch.Tensor:
    kwargs = {"size": (out_h, out_w), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    return F.interpolate(tensor, **kwargs)



def _resize_video_uint8_tchw(video_tchw_uint8: np.ndarray, out_h: int, out_w: int, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(video_tchw_uint8).float()
    resized = _interpolate_nchw(tensor, out_h, out_w, mode)
    return resized.round().clamp_(0, 255).to(torch.uint8)



def _resize_gray_uint8_thw(gray_thw_uint8: np.ndarray, out_h: int, out_w: int, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(gray_thw_uint8[:, None, :, :]).float()
    resized = _interpolate_nchw(tensor, out_h, out_w, mode)
    return resized[:, 0].round().clamp_(0, 255).to(torch.uint8)



def _pool_gray_to_grid_uint8(gray_thw_uint8: torch.Tensor, cell_size: int) -> torch.Tensor:
    if gray_thw_uint8.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape={tuple(gray_thw_uint8.shape)}")
    _, h, w = gray_thw_uint8.shape
    if h % cell_size != 0 or w % cell_size != 0:
        raise ValueError(f"H/W must be divisible by cell_size={cell_size}, got H={h}, W={w}")
    x = gray_thw_uint8.float().unsqueeze(1)
    y = F.avg_pool2d(x, kernel_size=cell_size, stride=cell_size)
    return y[:, 0].round().clamp_(0, 255).to(torch.uint8)



def _relative_output_path(output_dir: Path, input_dir: Path, video_path: Path) -> Path:
    rel = video_path.relative_to(input_dir)
    return output_dir / rel.with_suffix(".pt")


# -----------------------------
# Worker
# -----------------------------


def process_one_video(video_path_str: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = Config(**cfg_dict)
    video_path = Path(video_path_str)
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    out_path = _relative_output_path(output_dir, input_dir, video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoded_frames, decoded_video_fps = _decode_video_with_pyav(video_path)
    rgb_video, residual_gray, pict_types, times_sec, mv_stats = _build_original_residuals(decoded_frames)
    total_frames = int(rgb_video.shape[0])

    sample_idx, sampling_meta = hf_sample_frame_indices(
        total_num_frames=total_frames,
        metadata_fps=float(decoded_video_fps),
        cfg=cfg,
    )

    sampled_rgb = rgb_video[sample_idx]          # [T, H, W, 3]
    sampled_res = residual_gray[sample_idx]      # [T, H, W]
    sampled_ptypes = [pict_types[i] for i in sample_idx]
    sampled_times = [times_sec[i] for i in sample_idx]
    sampled_nframes = int(len(sample_idx))

    _, orig_h, orig_w, _ = sampled_rgb.shape
    resized_h, resized_w, resize_meta = compute_hf_video_resize(orig_h, orig_w, sampled_nframes, cfg)

    sampled_rgb_tchw = np.ascontiguousarray(sampled_rgb.transpose(0, 3, 1, 2))
    resized_video_uint8 = _resize_video_uint8_tchw(sampled_rgb_tchw, resized_h, resized_w, cfg.video_resize_mode)
    resized_residual_uint8 = _resize_gray_uint8_thw(sampled_res, resized_h, resized_w, cfg.residual_resize_mode)

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

    payload: Dict[str, Any] = {
        "metadata": {
            "video_path": str(video_path),
            "video_rel_path": str(video_path.relative_to(input_dir)),
            "original_total_frames": int(total_frames),
            "decoded_video_fps": float(decoded_video_fps),
            "sampled_nframes_before_temporal_pad": int(sampled_nframes),
            "sampled_indices": sample_idx,
            "sampled_times_sec": sampled_times,
            "sampled_pict_types": sampled_ptypes,
            "temporal_patch_size": int(cfg.temporal_patch_size),
            "temporal_pad_frames": int(temporal_pad),
            "temporal_padded_nframes": int(padded_nframes),
            "temporal_group_count": int(grid_t),
            "temporal_group_frame_indices": temporal_groups,
            "temporal_group_mid_times_sec": temporal_group_times_sec,
            "original_height": int(orig_h),
            "original_width": int(orig_w),
            "resized_height": int(resized_h),
            "resized_width": int(resized_w),
            "patch_size": int(cfg.patch_size),
            "spatial_merge_size": int(cfg.spatial_merge_size),
            "vision_patch_grid_hw": [vision_grid_h, vision_grid_w],
            "llm_grid_hw": [llm_grid_h, llm_grid_w],
            "hf_video_grid_thw": [grid_t, vision_grid_h, vision_grid_w],
            "hf_num_llm_tokens": int(grid_t * vision_grid_h * vision_grid_w // (cfg.spatial_merge_size ** 2)),
            "residual_definition": "abs(curr_luma - warp(prev_luma, mv_blocks_of_curr))",
            "mv_backend": "pyav_ffmpeg_side_data",
            "storage_mode": "compact_token_aligned_hf_video_processor",
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
            **mv_stats,
        }
    }

    if cfg.save_vision_residual_grid_uint8:
        payload["vision_residual_grid_uint8"] = _pool_gray_to_grid_uint8(
            resized_residual_uint8,
            cell_size=cfg.patch_size,
        )

    if cfg.save_llm_residual_grid_uint8:
        payload["llm_residual_grid_uint8"] = _pool_gray_to_grid_uint8(
            resized_residual_uint8,
            cell_size=llm_cell,
        )

    if cfg.save_residual_gray_uint8:
        payload["residual_gray_uint8"] = resized_residual_uint8

    if cfg.save_video_uint8:
        payload["video_uint8"] = resized_video_uint8

    if cfg.preview_frames > 0:
        p = min(int(cfg.preview_frames), int(sampled_nframes))
        payload["preview_video_uint8"] = resized_video_uint8[:p].clone()
        payload["preview_residual_gray_uint8"] = resized_residual_uint8[:p].clone()

    torch.save(payload, out_path)
    file_size = out_path.stat().st_size if out_path.exists() else 0

    return {
        "video_path": str(video_path),
        "output_path": str(out_path),
        "decoded_video_fps": round(float(decoded_video_fps), 6),
        "original_total_frames": int(total_frames),
        "sampled_nframes_before_temporal_pad": int(sampled_nframes),
        "temporal_padded_nframes": int(padded_nframes),
        "resized_height": int(resized_h),
        "resized_width": int(resized_w),
        "grid_t": int(grid_t),
        "vision_grid_h": int(vision_grid_h),
        "vision_grid_w": int(vision_grid_w),
        "llm_grid_h": int(llm_grid_h),
        "llm_grid_w": int(llm_grid_w),
        "num_i_frames": mv_stats["num_i_frames"],
        "num_p_frames": mv_stats["num_p_frames"],
        "num_frames_with_mv": mv_stats["num_frames_with_mv"],
        "num_frames_without_mv": mv_stats["num_frames_without_mv"],
        "output_bytes": int(file_size),
        "status": "ok",
    }


# -----------------------------
# Main / CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HF-aligned Qwen3-VL residual tensors from videos.")
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(os.cpu_count() or 4, 4))

    p.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--temporal_patch_size", type=int, default=DEFAULT_TEMPORAL_PATCH_SIZE)
    p.add_argument("--spatial_merge_size", type=int, default=DEFAULT_SPATIAL_MERGE_SIZE)
    p.add_argument("--processor_shortest_edge", type=int, default=DEFAULT_VIDEO_SHORTEST_EDGE)
    p.add_argument("--processor_longest_edge", type=int, default=DEFAULT_VIDEO_LONGEST_EDGE)
    p.add_argument("--processor_fps", type=float, default=DEFAULT_PROCESSOR_FPS)
    p.add_argument("--processor_min_frames", type=int, default=DEFAULT_PROCESSOR_MIN_FRAMES)
    p.add_argument("--processor_max_frames", type=int, default=DEFAULT_PROCESSOR_MAX_FRAMES)

    p.add_argument("--do_sample_frames", action="store_true", default=True,
                   help="Mimic HF processor-side sampling. Default: enabled.")
    p.add_argument("--no_do_sample_frames", action="store_true",
                   help="Disable HF processor-side sampling and keep all I/O-decoded frames.")
    p.add_argument("--sampling_fps", type=float, default=DEFAULT_PROCESSOR_FPS,
                   help="HF-side fps sampling target. Mutually exclusive with --sampling_num_frames.")
    p.add_argument("--sampling_num_frames", type=int, default=None,
                   help="HF-side explicit num_frames. Mutually exclusive with --sampling_fps.")
    p.add_argument("--runtime_max_frames", type=int, default=DEFAULT_RUNTIME_MAX_FRAMES,
                   help="Request-time max_frames you want to mimic, e.g. 2048.")
    p.add_argument("--runtime_max_frames_policy", type=str, default="override", choices=["ignore", "override", "min"],
                   help="How runtime max_frames combines with processor max_frames: ignore, override, or min.")

    p.add_argument("--save_video_uint8", action="store_true")
    p.add_argument("--save_residual_gray_uint8", action="store_true")
    p.add_argument("--no_save_vision_residual_grid_uint8", action="store_true")
    p.add_argument("--no_save_llm_residual_grid_uint8", action="store_true")
    p.add_argument("--preview_frames", type=int, default=0)
    p.add_argument("--residual_resize_mode", type=str, default="bicubic", choices=["nearest", "bilinear", "bicubic"])
    p.add_argument("--video_resize_mode", type=str, default="bicubic", choices=["nearest", "bilinear", "bicubic"])
    p.add_argument("--limit_videos", type=int, default=0)
    return p.parse_args()



def write_summary_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def main() -> int:
    args = parse_args()
    if av is None:
        print(
            "PyAV is not installed in the current environment. Please install it first, e.g. `pip install av`.",
            file=sys.stderr,
        )
        return 2

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if args.limit_videos and args.limit_videos > 0:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"No mp4 videos found under: {input_dir}", file=sys.stderr)
        return 1

    sampling_fps = None if args.sampling_num_frames is not None else args.sampling_fps
    do_sample_frames = False if args.no_do_sample_frames else bool(args.do_sample_frames)

    cfg = Config(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        workers=int(args.workers),
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
        save_video_uint8=bool(args.save_video_uint8),
        save_residual_gray_uint8=bool(args.save_residual_gray_uint8),
        save_vision_residual_grid_uint8=not bool(args.no_save_vision_residual_grid_uint8),
        save_llm_residual_grid_uint8=not bool(args.no_save_llm_residual_grid_uint8),
        preview_frames=int(args.preview_frames),
        residual_resize_mode=str(args.residual_resize_mode),
        video_resize_mode=str(args.video_resize_mode),
    )

    print(f"[INFO] Found {len(videos)} videos under {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
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

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(process_one_video, str(v), asdict(cfg)): v for v in videos}
        for fut in as_completed(futures):
            video = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"[OK] {video}")
            except Exception as exc:
                rows.append({
                    "video_path": str(video),
                    "output_path": "",
                    "decoded_video_fps": "",
                    "original_total_frames": "",
                    "sampled_nframes_before_temporal_pad": "",
                    "temporal_padded_nframes": "",
                    "resized_height": "",
                    "resized_width": "",
                    "grid_t": "",
                    "vision_grid_h": "",
                    "vision_grid_w": "",
                    "llm_grid_h": "",
                    "llm_grid_w": "",
                    "num_i_frames": "",
                    "num_p_frames": "",
                    "num_frames_with_mv": "",
                    "num_frames_without_mv": "",
                    "output_bytes": "",
                    "status": f"error: {exc}",
                })
                print(f"[ERR] {video}: {exc}", file=sys.stderr)

    rows.sort(key=lambda x: x["video_path"])
    write_summary_csv(output_dir / "summary.csv", rows)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"[DONE] succeeded={len(ok_rows)} / {len(rows)}")
    print(f"[DONE] summary: {output_dir / 'summary.csv'}")
    return 0 if ok_rows else 3


if __name__ == "__main__":
    raise SystemExit(main())
