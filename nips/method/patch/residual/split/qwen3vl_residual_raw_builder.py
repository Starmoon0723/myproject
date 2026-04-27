#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 builder:
- Decode video + motion vectors.
- Compute original residual maps only.
- Save raw residual tensor before any frame sampling / resize.

This script keeps the residual definition unchanged:
    abs(curr_luma - warp(prev_luma, mv_blocks_of_curr))
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qwen3vl_residual_tensor_builder_hf_aligned as core  # noqa: E402


@dataclass
class RawConfig:
    input_dir: str
    output_dir: str
    workers: int = min(max(os.cpu_count() or 4, 1), 4)
    limit_videos: int = 0
    resume: bool = True


def find_videos(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.mp4") if p.is_file()])


def _relative_output_path(output_dir: Path, input_dir: Path, video_path: Path) -> Path:
    rel = video_path.relative_to(input_dir)
    return output_dir / rel.with_suffix(".pt")


def _decode_compute_residual_stream(video_path: Path) -> Tuple[np.ndarray, List[str], List[float], Dict[str, int], float]:
    av_mod = core._require_pyav()
    residuals: List[np.ndarray] = []
    pict_types: List[str] = []
    times_sec: List[float] = []
    stats = {
        "num_i_frames": 0,
        "num_p_frames": 0,
        "num_other_frames": 0,
        "num_frames_with_mv": 0,
        "num_frames_without_mv": 0,
    }

    prev_luma: Optional[np.ndarray] = None
    decoded_video_fps = 0.0

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
        decoded_video_fps = float(avg_rate) if avg_rate is not None else 0.0

        frame_idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                rgb = frame.to_ndarray(format="rgb24")
                curr_luma = core._rgb_to_luma_uint8(rgb)
                time_sec = float(core._frame_time_seconds(frame, stream))
                pict_type = core._frame_pict_type(frame)
                ptype = pict_type.upper() if pict_type is not None else "?"

                pict_types.append(pict_type)
                times_sec.append(time_sec)

                if ptype == "I":
                    stats["num_i_frames"] += 1
                elif ptype == "P":
                    stats["num_p_frames"] += 1
                else:
                    stats["num_other_frames"] += 1

                if frame_idx == 0 or ptype == "I":
                    residual = np.zeros_like(curr_luma, dtype=np.uint8)
                else:
                    mv_side_data = core._get_mv_side_data(frame)
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

                    if mv_blocks:
                        stats["num_frames_with_mv"] += 1
                        pred = core._warp_prev_luma_with_mvs(prev_luma, mv_blocks)
                    else:
                        stats["num_frames_without_mv"] += 1
                        pred = prev_luma

                    residual = np.abs(curr_luma.astype(np.int16) - pred.astype(np.int16)).astype(np.uint8)

                residuals.append(residual)
                prev_luma = curr_luma
                frame_idx += 1

    if not residuals:
        raise ValueError("No decoded frames found.")

    if decoded_video_fps <= 0 and len(times_sec) >= 2:
        duration = max(times_sec[-1] - times_sec[0], 1e-6)
        decoded_video_fps = (len(times_sec) - 1) / duration
    if decoded_video_fps <= 0:
        decoded_video_fps = core.DEFAULT_PROCESSOR_FPS

    raw_residual = np.stack(residuals, axis=0)
    return raw_residual, pict_types, times_sec, stats, float(decoded_video_fps)


def process_one_video(video_path_str: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    cfg = RawConfig(**cfg_dict)
    video_path = Path(video_path_str)
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    out_path = _relative_output_path(output_dir, input_dir, video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_residual, pict_types, times_sec, mv_stats, decoded_video_fps = _decode_compute_residual_stream(video_path)
    _, h, w = raw_residual.shape
    total_frames = int(raw_residual.shape[0])

    payload: Dict[str, Any] = {
        "metadata": {
            "video_path": str(video_path),
            "video_rel_path": str(video_path.relative_to(input_dir)),
            "original_total_frames": int(total_frames),
            "decoded_video_fps": float(decoded_video_fps),
            "original_height": int(h),
            "original_width": int(w),
            "frame_times_sec": times_sec,
            "frame_pict_types": pict_types,
            "residual_definition": "abs(curr_luma - warp(prev_luma, mv_blocks_of_curr))",
            "mv_backend": "pyav_ffmpeg_side_data",
            "storage_mode": "raw_residual_before_sampling_resize",
            **mv_stats,
        },
        "raw_residual_gray_uint8": torch.from_numpy(raw_residual).to(torch.uint8),
    }
    torch.save(payload, out_path)
    file_size = out_path.stat().st_size if out_path.exists() else 0

    return {
        "video_path": str(video_path),
        "output_path": str(out_path),
        "decoded_video_fps": round(float(decoded_video_fps), 6),
        "original_total_frames": int(total_frames),
        "original_height": int(h),
        "original_width": int(w),
        "num_i_frames": mv_stats["num_i_frames"],
        "num_p_frames": mv_stats["num_p_frames"],
        "num_frames_with_mv": mv_stats["num_frames_with_mv"],
        "num_frames_without_mv": mv_stats["num_frames_without_mv"],
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
    p = argparse.ArgumentParser(description="Stage-1: build raw residual tensors before sampling/resize.")
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=min(max(os.cpu_count() or 4, 1), 4))
    p.add_argument("--limit_videos", type=int, default=0)
    p.add_argument("--force_recompute", action="store_true", help="Recompute even if output .pt already exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if core.av is None:
        print("PyAV is not installed. Please install it first, e.g. `pip install av`.", file=sys.stderr)
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

    cfg = RawConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        workers=max(1, int(args.workers)),
        limit_videos=int(args.limit_videos),
        resume=not bool(args.force_recompute),
    )

    print(f"[INFO] Found {len(videos)} videos under {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] workers={cfg.workers}")
    print(f"[INFO] resume={cfg.resume}")

    rows: List[Dict[str, Any]] = []
    pending_videos: List[Path] = []
    for video in videos:
        out_path = _relative_output_path(output_dir, input_dir, video)
        if cfg.resume and out_path.exists():
            rows.append(
                {
                    "video_path": str(video),
                    "output_path": str(out_path),
                    "decoded_video_fps": "",
                    "original_total_frames": "",
                    "original_height": "",
                    "original_width": "",
                    "num_i_frames": "",
                    "num_p_frames": "",
                    "num_frames_with_mv": "",
                    "num_frames_without_mv": "",
                    "output_bytes": int(out_path.stat().st_size),
                    "status": "skipped_existing",
                }
            )
            continue
        pending_videos.append(video)

    print(f"[INFO] to_process={len(pending_videos)}, skipped_existing={len(rows)}")

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(process_one_video, str(v), asdict(cfg)): v for v in pending_videos}
        for fut in as_completed(futures):
            video = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"[OK] {video}")
            except Exception as exc:
                rows.append(
                    {
                        "video_path": str(video),
                        "output_path": "",
                        "decoded_video_fps": "",
                        "original_total_frames": "",
                        "original_height": "",
                        "original_width": "",
                        "num_i_frames": "",
                        "num_p_frames": "",
                        "num_frames_with_mv": "",
                        "num_frames_without_mv": "",
                        "output_bytes": "",
                        "status": f"error: {exc}",
                    }
                )
                print(f"[ERR] {video}: {exc}", file=sys.stderr)

    rows.sort(key=lambda x: x["video_path"])
    write_summary_csv(output_dir / "summary.csv", rows)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    skipped_rows = [r for r in rows if r.get("status") == "skipped_existing"]
    print(f"[DONE] succeeded={len(ok_rows)} / {len(rows)} (skipped={len(skipped_rows)})")
    print(f"[DONE] summary: {output_dir / 'summary.csv'}")
    return 0 if ok_rows or skipped_rows else 3


if __name__ == "__main__":
    raise SystemExit(main())
