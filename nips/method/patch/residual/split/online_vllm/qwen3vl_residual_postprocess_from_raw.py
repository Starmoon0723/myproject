#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 postprocess:
- Read raw residual tensors produced by stage-1.
- Apply HF-aligned frame sampling + smart resize.
- Save compact residual outputs.
"""

from __future__ import annotations

import argparse
import csv
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
class PostConfig:
    input_dir: str
    output_dir: str
    workers: int = min(max(os.cpu_count() or 4, 1), 4)

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

    save_residual_gray_uint8: bool = False
    save_vision_residual_grid_uint8: bool = True
    save_llm_residual_grid_uint8: bool = True
    preview_frames: int = 0
    residual_resize_mode: str = "bicubic"
    limit_tensors: int = 0


def _to_core_config(cfg: PostConfig) -> core.Config:
    return core.Config(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir,
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
        save_residual_gray_uint8=cfg.save_residual_gray_uint8,
        save_vision_residual_grid_uint8=cfg.save_vision_residual_grid_uint8,
        save_llm_residual_grid_uint8=cfg.save_llm_residual_grid_uint8,
        preview_frames=cfg.preview_frames,
        residual_resize_mode=cfg.residual_resize_mode,
        video_resize_mode="bicubic",
    )


def find_raw_tensors(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.pt") if p.is_file()])


def _relative_output_path(output_dir: Path, input_dir: Path, raw_path: Path) -> Path:
    rel = raw_path.relative_to(input_dir)
    return output_dir / rel


def _build_fallback_frame_times(total_frames: int, fps: float) -> List[float]:
    if fps <= 0:
        fps = core.DEFAULT_PROCESSOR_FPS
    return [float(i / fps) for i in range(total_frames)]


def process_one_raw_tensor(raw_path_str: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = PostConfig(**cfg_dict)
    core_cfg = _to_core_config(cfg)

    raw_path = Path(raw_path_str)
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    out_path = _relative_output_path(output_dir, input_dir, raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_payload = torch.load(raw_path, map_location="cpu")
    if "raw_residual_gray_uint8" not in raw_payload:
        raise ValueError("Input tensor does not contain `raw_residual_gray_uint8` key.")

    raw_residual = raw_payload["raw_residual_gray_uint8"]
    if not isinstance(raw_residual, torch.Tensor):
        raise ValueError("`raw_residual_gray_uint8` must be a torch.Tensor.")
    if raw_residual.ndim != 3:
        raise ValueError(f"`raw_residual_gray_uint8` must be [T,H,W], got {tuple(raw_residual.shape)}.")

    raw_residual = raw_residual.to(torch.uint8).contiguous().cpu()
    total_frames = int(raw_residual.shape[0])
    orig_h = int(raw_residual.shape[1])
    orig_w = int(raw_residual.shape[2])
    if total_frames <= 0:
        raise ValueError("No frames found in raw residual tensor.")

    raw_meta = raw_payload.get("metadata", {}) or {}
    decoded_video_fps = float(raw_meta.get("decoded_video_fps", core.DEFAULT_PROCESSOR_FPS))
    frame_times_sec = list(raw_meta.get("frame_times_sec", []))
    frame_pict_types = list(raw_meta.get("frame_pict_types", []))

    if len(frame_times_sec) != total_frames:
        frame_times_sec = _build_fallback_frame_times(total_frames, decoded_video_fps)
    if len(frame_pict_types) != total_frames:
        frame_pict_types = ["?"] * total_frames

    sample_idx, sampling_meta = core.hf_sample_frame_indices(
        total_num_frames=total_frames,
        metadata_fps=decoded_video_fps,
        cfg=core_cfg,
    )

    sampled_residual = raw_residual[sample_idx]
    sampled_nframes = int(sampled_residual.shape[0])
    sampled_times = [frame_times_sec[i] for i in sample_idx]
    sampled_ptypes = [frame_pict_types[i] for i in sample_idx]

    resized_h, resized_w, resize_meta = core.compute_hf_video_resize(orig_h, orig_w, sampled_nframes, core_cfg)
    sampled_residual_np = sampled_residual.numpy()
    resized_residual_uint8 = core._resize_gray_uint8_thw(
        sampled_residual_np,
        resized_h,
        resized_w,
        cfg.residual_resize_mode,
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

    payload: Dict[str, Any] = {
        "metadata": {
            "source_raw_path": str(raw_path),
            "source_video_path": str(raw_meta.get("video_path", "")),
            "source_video_rel_path": str(raw_meta.get("video_rel_path", "")),
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
            "residual_definition": raw_meta.get(
                "residual_definition",
                "abs(curr_luma - warp(prev_luma, mv_blocks_of_curr))",
            ),
            "mv_backend": raw_meta.get("mv_backend", "pyav_ffmpeg_side_data"),
            "storage_mode": "compact_token_aligned_hf_video_processor_from_raw_residual",
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
            "num_i_frames": int(raw_meta.get("num_i_frames", 0)),
            "num_p_frames": int(raw_meta.get("num_p_frames", 0)),
            "num_other_frames": int(raw_meta.get("num_other_frames", 0)),
            "num_frames_with_mv": int(raw_meta.get("num_frames_with_mv", 0)),
            "num_frames_without_mv": int(raw_meta.get("num_frames_without_mv", 0)),
        }
    }

    if cfg.save_vision_residual_grid_uint8:
        payload["vision_residual_grid_uint8"] = core._pool_gray_to_grid_uint8(
            resized_residual_uint8,
            cell_size=cfg.patch_size,
        )

    if cfg.save_llm_residual_grid_uint8:
        payload["llm_residual_grid_uint8"] = core._pool_gray_to_grid_uint8(
            resized_residual_uint8,
            cell_size=llm_cell,
        )

    if cfg.save_residual_gray_uint8:
        payload["residual_gray_uint8"] = resized_residual_uint8

    if cfg.preview_frames > 0:
        p = min(int(cfg.preview_frames), int(sampled_nframes))
        payload["preview_residual_gray_uint8"] = resized_residual_uint8[:p].clone()

    torch.save(payload, out_path)
    file_size = out_path.stat().st_size if out_path.exists() else 0

    return {
        "raw_path": str(raw_path),
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
        "num_i_frames": int(raw_meta.get("num_i_frames", 0)),
        "num_p_frames": int(raw_meta.get("num_p_frames", 0)),
        "num_frames_with_mv": int(raw_meta.get("num_frames_with_mv", 0)),
        "num_frames_without_mv": int(raw_meta.get("num_frames_without_mv", 0)),
        "output_bytes": int(file_size),
        "status": "ok",
    }


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2: postprocess raw residual tensors with HF-aligned sampling/resize.")
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
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

    p.add_argument("--save_residual_gray_uint8", action="store_true")
    p.add_argument("--no_save_vision_residual_grid_uint8", action="store_true")
    p.add_argument("--no_save_llm_residual_grid_uint8", action="store_true")
    p.add_argument("--preview_frames", type=int, default=0)
    p.add_argument("--residual_resize_mode", type=str, default="bicubic", choices=["nearest", "bilinear", "bicubic"])
    p.add_argument("--limit_tensors", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = find_raw_tensors(input_dir)
    if args.limit_tensors and args.limit_tensors > 0:
        tensors = tensors[: args.limit_tensors]
    if not tensors:
        print(f"No .pt files found under: {input_dir}", file=sys.stderr)
        return 1

    sampling_fps = None if args.sampling_num_frames is not None else args.sampling_fps
    do_sample_frames = False if args.no_do_sample_frames else bool(args.do_sample_frames)

    cfg = PostConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
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
        save_residual_gray_uint8=bool(args.save_residual_gray_uint8),
        save_vision_residual_grid_uint8=not bool(args.no_save_vision_residual_grid_uint8),
        save_llm_residual_grid_uint8=not bool(args.no_save_llm_residual_grid_uint8),
        preview_frames=int(args.preview_frames),
        residual_resize_mode=str(args.residual_resize_mode),
        limit_tensors=int(args.limit_tensors),
    )

    print(f"[INFO] Found {len(tensors)} raw tensors under {input_dir}")
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
        futures = {ex.submit(process_one_raw_tensor, str(p), asdict(cfg)): p for p in tensors}
        for fut in as_completed(futures):
            source = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"[OK] {source}")
            except Exception as exc:
                rows.append(
                    {
                        "raw_path": str(source),
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
                    }
                )
                print(f"[ERR] {source}: {exc}", file=sys.stderr)

    rows.sort(key=lambda x: x.get("raw_path", ""))
    write_summary_csv(output_dir / "summary.csv", rows)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"[DONE] succeeded={len(ok_rows)} / {len(rows)}")
    print(f"[DONE] summary: {output_dir / 'summary.csv'}")
    return 0 if ok_rows else 3


if __name__ == "__main__":
    raise SystemExit(main())
