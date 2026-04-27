#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize token selection results from qwen3vl_residual_token_selector.py.

Output layout:
  <output_dir>/
    <video_name_or_index>/
      summary.json
      group_0000/
        frame_00_original.jpg
        frame_00_grid.jpg
        frame_00_selected.jpg
        frame_00_mask.png
        ...
      group_0001/
        ...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_video_name(row: Dict[str, Any], idx: int) -> str:
    if "index" in row:
        base = f"idx_{row['index']}"
    else:
        base = f"sample_{idx:04d}"
    rel = row.get("source_video_rel_path", "") or row.get("source_video_path", "")
    if rel:
        stem = Path(str(rel)).stem
        return f"{base}_{stem}"
    return base


def _resolve_tensor_path(
    row: Dict[str, Any],
    tensor_root: Optional[Path],
) -> Path:
    p = row.get("source_tensor_path", None)
    if p is None:
        raise ValueError("Row missing `source_tensor_path`.")
    path = Path(str(p))
    if path.exists():
        return path
    if tensor_root is not None:
        candidate = tensor_root / path.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Tensor not found: {path}")


def _resolve_video_path(
    row: Dict[str, Any],
    metadata: Dict[str, Any],
    video_root: Optional[Path],
) -> Path:
    candidates = [
        row.get("source_video_path", ""),
        metadata.get("source_video_path", ""),
    ]
    for p in candidates:
        if p:
            path = Path(str(p))
            if path.exists():
                return path
    rel_candidates = [
        row.get("source_video_rel_path", ""),
        metadata.get("source_video_rel_path", ""),
    ]
    if video_root is not None:
        for rel in rel_candidates:
            if rel:
                path = video_root / str(rel)
                if path.exists():
                    return path
    raise FileNotFoundError(
        f"Video not found. candidates={candidates}, rel_candidates={rel_candidates}, video_root={video_root}"
    )


def _decode_frames_by_index(video_path: Path, frame_indices: List[int]) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    out: Dict[int, np.ndarray] = {}
    unique_indices = sorted(set(int(i) for i in frame_indices))
    for idx in unique_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out[idx] = frame_rgb
    cap.release()
    return out


def _build_mask_image(
    keep_indices: List[int],
    llm_grid_h: int,
    llm_grid_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    keep_set = set(int(x) for x in keep_indices)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cell_h = out_h // llm_grid_h
    cell_w = out_w // llm_grid_w
    for r in range(llm_grid_h):
        for c in range(llm_grid_w):
            token_idx = r * llm_grid_w + c
            if token_idx in keep_set:
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                mask[y0:y1, x0:x1] = 255
    return mask


def _draw_grid(img: Image.Image, llm_grid_h: int, llm_grid_w: int, color: Tuple[int, int, int]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cell_h = h // llm_grid_h
    cell_w = w // llm_grid_w
    for r in range(llm_grid_h + 1):
        y = min(r * cell_h, h - 1)
        draw.line([(0, y), (w - 1, y)], fill=color, width=1)
    for c in range(llm_grid_w + 1):
        x = min(c * cell_w, w - 1)
        draw.line([(x, 0), (x, h - 1)], fill=color, width=1)
    return out


def _draw_selection_overlay(
    img: Image.Image,
    keep_indices: List[int],
    llm_grid_h: int,
    llm_grid_w: int,
) -> Image.Image:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    keep_set = set(int(x) for x in keep_indices)
    w, h = base.size
    cell_h = h // llm_grid_h
    cell_w = w // llm_grid_w
    for r in range(llm_grid_h):
        for c in range(llm_grid_w):
            token_idx = r * llm_grid_w + c
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            if token_idx in keep_set:
                draw.rectangle((x0, y0, x1, y1), fill=(0, 220, 0, 60), outline=(0, 180, 0, 255), width=1)
            else:
                draw.rectangle((x0, y0, x1, y1), fill=(220, 0, 0, 35))
    merged = Image.alpha_composite(base, overlay).convert("RGB")
    return _draw_grid(merged, llm_grid_h, llm_grid_w, color=(255, 255, 255))


def _process_one_video(
    row: Dict[str, Any],
    out_dir: Path,
    *,
    tensor_root: Optional[Path],
    video_root: Optional[Path],
    max_groups: int,
) -> None:
    tensor_path = _resolve_tensor_path(row, tensor_root)
    payload = torch.load(tensor_path, map_location="cpu")
    metadata = payload.get("metadata", {}) or {}
    video_path = _resolve_video_path(row, metadata, video_root)

    sampled_indices = list(metadata.get("sampled_indices", []))
    temporal_patch_size = int(metadata.get("temporal_patch_size", 2))
    temporal_pad = int(metadata.get("temporal_pad_frames", 0))
    resized_h = int(metadata.get("resized_height"))
    resized_w = int(metadata.get("resized_width"))
    llm_grid_hw = metadata.get("llm_grid_hw", None)
    if not isinstance(llm_grid_hw, (list, tuple)) or len(llm_grid_hw) != 2:
        raise ValueError(f"Invalid llm_grid_hw in metadata: {llm_grid_hw}")
    llm_grid_h = int(llm_grid_hw[0])
    llm_grid_w = int(llm_grid_hw[1])

    keep_groups = row.get("video_token_keep_indices", None)
    if keep_groups is None:
        raise ValueError("Row missing `video_token_keep_indices`.")
    keep_groups = [list(map(int, g)) for g in keep_groups]

    sampled_indices_padded = sampled_indices + ([sampled_indices[-1]] * temporal_pad if temporal_pad > 0 else [])
    decoded = _decode_frames_by_index(video_path, sampled_indices_padded)

    group_count = len(keep_groups)
    if max_groups > 0:
        group_count = min(group_count, max_groups)

    summary = {
        "source_video_path": str(video_path),
        "source_tensor_path": str(tensor_path),
        "sampled_indices_len": len(sampled_indices),
        "temporal_patch_size": temporal_patch_size,
        "temporal_pad": temporal_pad,
        "resized_hw": [resized_h, resized_w],
        "llm_grid_hw": [llm_grid_h, llm_grid_w],
        "group_count_visualized": group_count,
        "group_count_total": len(keep_groups),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for gid in range(group_count):
        group_dir = out_dir / f"group_{gid:04d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        keep_indices = keep_groups[gid]
        st = gid * temporal_patch_size
        ed = st + temporal_patch_size
        group_frame_indices = sampled_indices_padded[st:ed]
        if len(group_frame_indices) == 0:
            continue

        for local_fid, frame_idx in enumerate(group_frame_indices):
            frame = decoded.get(int(frame_idx), None)
            if frame is None:
                continue
            frame_resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
            pil_img = Image.fromarray(frame_resized)

            original_path = group_dir / f"frame_{local_fid:02d}_original.jpg"
            grid_path = group_dir / f"frame_{local_fid:02d}_grid.jpg"
            selected_path = group_dir / f"frame_{local_fid:02d}_selected.jpg"
            mask_path = group_dir / f"frame_{local_fid:02d}_mask.png"

            pil_img.save(original_path, quality=95)
            _draw_grid(pil_img, llm_grid_h, llm_grid_w, color=(0, 255, 255)).save(grid_path, quality=95)
            _draw_selection_overlay(pil_img, keep_indices, llm_grid_h, llm_grid_w).save(selected_path, quality=95)
            mask = _build_mask_image(keep_indices, llm_grid_h, llm_grid_w, resized_h, resized_w)
            Image.fromarray(mask).save(mask_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize selected token indices for Qwen3VL residual pruning.")
    p.add_argument("--selected_jsonl", type=Path, required=True, help="Path to selected_token.jsonl")
    p.add_argument("--output_dir", type=Path, required=True, help="Visualization output directory")
    p.add_argument("--num_videos", type=int, default=5, help="How many videos to visualize")
    p.add_argument("--max_groups", type=int, default=0, help="Max temporal groups per video (0 means all)")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--random_pick", action="store_true", help="Randomly sample videos instead of first N")
    p.add_argument("--tensor_root", type=Path, default=None, help="Optional fallback tensor root directory")
    p.add_argument("--video_root", type=Path, default=None, help="Optional fallback video root directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.selected_jsonl)
    if len(rows) == 0:
        raise ValueError(f"Empty jsonl: {args.selected_jsonl}")

    if args.random_pick:
        random.seed(args.seed)
        pick = min(args.num_videos, len(rows))
        rows = random.sample(rows, k=pick)
    else:
        rows = rows[: min(args.num_videos, len(rows))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        name = _safe_video_name(row, i)
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _process_one_video(
                row,
                out_dir,
                tensor_root=args.tensor_root,
                video_root=args.video_root,
                max_groups=int(args.max_groups),
            )
            print(f"[OK] {name}")
        except Exception as exc:
            err_path = out_dir / "error.txt"
            err_path.write_text(str(exc), encoding="utf-8")
            print(f"[ERR] {name}: {exc}")

    print(f"[DONE] output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

