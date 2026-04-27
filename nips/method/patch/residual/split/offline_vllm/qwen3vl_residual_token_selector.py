#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-3 token selector (offline-vLLM aligned):
- Read stage-2 residual outputs (llm_residual_grid_uint8 + metadata).
- Select per-temporal-group token indices under a target global delete ratio.
- Keep I-anchor groups unpruned by default (plus periodic GOP anchors).
- Export JSONL for downstream vLLM `video_token_keep_indices` injection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _norm_path(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lower()


def _find_pt_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.pt") if p.is_file()])


def _load_dataset_index_map(
    dataset_tsv: Optional[Path],
    *,
    video_path_col: str,
    index_col: str,
    data_root: Optional[Path],
) -> Dict[str, int]:
    if dataset_tsv is None:
        return {}
    mapping: Dict[str, int] = {}
    with dataset_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if video_path_col not in row or index_col not in row:
                continue
            rel_path = row[video_path_col]
            try:
                sample_idx = int(row[index_col])
            except Exception:
                continue
            rel_norm = _norm_path(rel_path)
            mapping[rel_norm] = sample_idx
            if data_root is not None:
                abs_norm = _norm_path(str((data_root / rel_path).resolve()))
                mapping[abs_norm] = sample_idx
    return mapping


def _build_anchor_groups(
    metadata: Dict[str, Any],
    *,
    temporal_patch_size: int,
    group_count: int,
    keep_i_groups: bool,
    gop_seconds: float,
) -> set[int]:
    anchors: set[int] = set()

    if keep_i_groups:
        sampled_ptypes = [str(x).upper() for x in metadata.get("sampled_pict_types", [])]
        if len(sampled_ptypes) > 0:
            temporal_pad = int(metadata.get("temporal_pad_frames", 0))
            if temporal_pad > 0:
                sampled_ptypes = sampled_ptypes + [sampled_ptypes[-1]] * temporal_pad
            for gid in range(group_count):
                st = gid * temporal_patch_size
                ed = st + temporal_patch_size
                ptypes = sampled_ptypes[st:ed]
                if any(pt == "I" for pt in ptypes):
                    anchors.add(gid)

    if gop_seconds > 0:
        mids = metadata.get("temporal_group_mid_times_sec", []) or []
        first_gid_by_gop: Dict[int, int] = {}
        for gid, mid_t in enumerate(mids[:group_count]):
            try:
                gop_id = int(math.floor(float(mid_t) / gop_seconds))
            except Exception:
                continue
            if gop_id not in first_gid_by_gop:
                first_gid_by_gop[gop_id] = gid
        anchors.update(first_gid_by_gop.values())

    if len(anchors) == 0 and group_count > 0:
        anchors.add(0)
    return anchors


def _select_indices_for_one_video(
    payload: Dict[str, Any],
    *,
    delete_ratio: float,
    keep_i_groups: bool,
    gop_seconds: float,
) -> Dict[str, Any]:
    if "llm_residual_grid_uint8" not in payload:
        raise ValueError("Missing key `llm_residual_grid_uint8` in payload.")
    metadata = payload.get("metadata", {}) or {}
    llm_grid = payload["llm_residual_grid_uint8"]
    if not isinstance(llm_grid, torch.Tensor) or llm_grid.ndim != 3:
        raise ValueError("`llm_residual_grid_uint8` must be a tensor with shape [T,H,W].")

    llm_grid = llm_grid.to(torch.float32).contiguous().cpu()
    sampled_nframes = int(llm_grid.shape[0])
    tokens_per_group = int(llm_grid.shape[1] * llm_grid.shape[2])
    temporal_patch_size = int(metadata.get("temporal_patch_size", 2))
    temporal_pad = int(metadata.get("temporal_pad_frames", 0))
    padded_nframes = int(sampled_nframes + temporal_pad)
    group_count = max(padded_nframes // temporal_patch_size, 1)

    if temporal_pad > 0:
        pad_frames = llm_grid[-1:].repeat(temporal_pad, 1, 1)
        llm_grid = torch.cat([llm_grid, pad_frames], dim=0)
    llm_flat = llm_grid.view(padded_nframes, tokens_per_group)
    grouped = llm_flat.view(group_count, temporal_patch_size, tokens_per_group)
    group_scores = grouped.max(dim=1).values

    anchors = _build_anchor_groups(
        metadata,
        temporal_patch_size=temporal_patch_size,
        group_count=group_count,
        keep_i_groups=keep_i_groups,
        gop_seconds=gop_seconds,
    )

    total_tokens = group_count * tokens_per_group
    target_keep = int(round(total_tokens * (1.0 - delete_ratio)))
    target_keep = max(0, min(target_keep, total_tokens))
    forced_keep = len(anchors) * tokens_per_group

    keep_indices: List[List[int]] = [[] for _ in range(group_count)]
    for gid in anchors:
        keep_indices[gid] = list(range(tokens_per_group))

    non_anchor_groups = [gid for gid in range(group_count) if gid not in anchors]
    non_anchor_capacity = len(non_anchor_groups) * tokens_per_group
    budget_non_anchor = max(0, min(target_keep - forced_keep, non_anchor_capacity))

    if budget_non_anchor > 0 and len(non_anchor_groups) > 0:
        non_anchor_scores = group_scores[non_anchor_groups, :].reshape(-1)
        topk = torch.topk(non_anchor_scores, k=budget_non_anchor, largest=True).indices
        for flat_idx in topk.tolist():
            local_gid = int(flat_idx // tokens_per_group)
            token_idx = int(flat_idx % tokens_per_group)
            gid = non_anchor_groups[local_gid]
            keep_indices[gid].append(token_idx)
        for gid in non_anchor_groups:
            keep_indices[gid] = sorted(set(keep_indices[gid]))

    actual_keep = sum(len(x) for x in keep_indices)
    actual_delete_ratio = 1.0 - (actual_keep / max(total_tokens, 1))

    return {
        "video_token_keep_indices": keep_indices,
        "tokens_per_group": tokens_per_group,
        "group_count": group_count,
        "anchor_group_count": len(anchors),
        "target_delete_ratio": float(delete_ratio),
        "actual_delete_ratio": float(actual_delete_ratio),
        "total_tokens": int(total_tokens),
        "kept_tokens": int(actual_keep),
    }


@dataclass
class SelectorConfig:
    input_dir: str
    output_jsonl: str
    delete_ratio: float
    keep_i_groups: bool
    gop_seconds: float
    dataset_tsv: Optional[str] = None
    video_path_col: str = "video_path"
    index_col: str = "index"
    data_root: Optional[str] = None
    limit_files: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline residual token selector for Qwen3-VL video pruning.")
    p.add_argument("--input_dir", type=Path, required=True, help="Directory of stage-2 .pt outputs.")
    p.add_argument("--output_jsonl", type=Path, required=True, help="Output manifest JSONL path.")
    p.add_argument("--delete_ratio", type=float, required=True, help="Target global deletion ratio in [0,1).")
    p.add_argument("--keep_i_groups", action="store_true", default=True)
    p.add_argument("--no_keep_i_groups", action="store_true")
    p.add_argument("--gop_seconds", type=float, default=8.0, help="Anchor first temporal group in each GOP window.")
    p.add_argument("--dataset_tsv", type=Path, default=None, help="Optional Video-MME TSV for writing sample index.")
    p.add_argument("--video_path_col", type=str, default="video_path")
    p.add_argument("--index_col", type=str, default="index")
    p.add_argument("--data_root", type=Path, default=None, help="Optional dataset root for abs path index mapping.")
    p.add_argument("--limit_files", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    keep_i_groups = False if args.no_keep_i_groups else bool(args.keep_i_groups)
    cfg = SelectorConfig(
        input_dir=str(args.input_dir),
        output_jsonl=str(args.output_jsonl),
        delete_ratio=float(args.delete_ratio),
        keep_i_groups=keep_i_groups,
        gop_seconds=float(args.gop_seconds),
        dataset_tsv=None if args.dataset_tsv is None else str(args.dataset_tsv),
        video_path_col=str(args.video_path_col),
        index_col=str(args.index_col),
        data_root=None if args.data_root is None else str(args.data_root),
        limit_files=int(args.limit_files),
    )

    if not (0.0 <= cfg.delete_ratio < 1.0):
        raise ValueError(f"delete_ratio must be in [0,1), got {cfg.delete_ratio}.")

    input_dir = Path(cfg.input_dir)
    output_jsonl = Path(cfg.output_jsonl)
    pt_files = _find_pt_files(input_dir)
    if cfg.limit_files > 0:
        pt_files = pt_files[: cfg.limit_files]
    if len(pt_files) == 0:
        raise ValueError(f"No .pt files found under {input_dir}")

    index_map = _load_dataset_index_map(
        None if cfg.dataset_tsv is None else Path(cfg.dataset_tsv),
        video_path_col=cfg.video_path_col,
        index_col=cfg.index_col,
        data_root=None if cfg.data_root is None else Path(cfg.data_root),
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for i, pt_path in enumerate(pt_files, start=1):
        payload = torch.load(pt_path, map_location="cpu")
        metadata = payload.get("metadata", {}) or {}
        selection = _select_indices_for_one_video(
            payload,
            delete_ratio=cfg.delete_ratio,
            keep_i_groups=cfg.keep_i_groups,
            gop_seconds=cfg.gop_seconds,
        )
        rel_path = str(metadata.get("source_video_rel_path", ""))
        abs_path = str(metadata.get("source_video_path", ""))
        norm_candidates = [_norm_path(rel_path), _norm_path(abs_path)]
        sample_index = None
        for key in norm_candidates:
            if key in index_map:
                sample_index = index_map[key]
                break

        row: Dict[str, Any] = {
            "source_video_rel_path": rel_path,
            "source_video_path": abs_path,
            "source_tensor_path": str(pt_path),
            "video_token_keep_indices": selection["video_token_keep_indices"],
            "tokens_per_group": selection["tokens_per_group"],
            "group_count": selection["group_count"],
            "anchor_group_count": selection["anchor_group_count"],
            "target_delete_ratio": selection["target_delete_ratio"],
            "actual_delete_ratio": selection["actual_delete_ratio"],
            "total_tokens": selection["total_tokens"],
            "kept_tokens": selection["kept_tokens"],
        }
        if sample_index is not None:
            row[cfg.index_col] = sample_index
        rows.append(row)

        if i % 100 == 0:
            print(f"[INFO] processed {i}/{len(pt_files)}")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg_delete_ratio = sum(r["actual_delete_ratio"] for r in rows) / max(len(rows), 1)
    print(f"[DONE] wrote {len(rows)} rows to {output_jsonl}")
    print(f"[DONE] target_delete_ratio={cfg.delete_ratio:.4f}, avg_actual_delete_ratio={avg_delete_ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
