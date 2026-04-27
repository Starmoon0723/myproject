#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-3 token selector (offline-vLLM aligned, EVS-style budget allocation).

Design goals
------------
1) Keep the pruning budget allocation closer to upstream EVS:
   - global ranking across all non-anchor tokens
   - stable argsort instead of topk
   - only keep the first temporal group fully by default
2) Make residual-based scoring more robust than a pure temporal max:
   - temporal mean/max/std hybrid aggregation
   - optional spatial smoothing to suppress isolated noisy spikes
3) Avoid catastrophic temporal starvation that is common when using
   residual-only global ranking:
   - optional tiny per-group keep floor for non-anchor groups
4) Preserve the JSONL export format used by downstream custom vLLM injection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------------
# I/O helpers
# -----------------------------


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


# -----------------------------
# Score loading / aggregation
# -----------------------------


def _as_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if x is None:
        return None
    try:
        return torch.as_tensor(x)
    except Exception:
        return None


def _load_score_grid(payload: Dict[str, Any], preferred_key: str = "auto") -> Tuple[torch.Tensor, str]:
    if preferred_key != "auto":
        if preferred_key not in payload:
            raise ValueError(f"preferred score key `{preferred_key}` not found in payload")
        tensor = _as_tensor(payload[preferred_key])
        if tensor is None:
            raise ValueError(f"score key `{preferred_key}` cannot be converted to tensor")
        if tensor.ndim != 3:
            raise ValueError(f"score key `{preferred_key}` must have shape [T,H,W], got {tuple(tensor.shape)}")
        return tensor.to(torch.float32).contiguous().cpu(), preferred_key

    candidate_keys = [
        # higher-precision variants first
        "llm_residual_grid_fp32",
        "llm_residual_grid_float32",
        "llm_residual_grid_fp16",
        "llm_residual_grid_float16",
        "llm_residual_grid_float",
        "llm_residual_grid",
        # lowest precision fallback
        "llm_residual_grid_uint8",
    ]
    for key in candidate_keys:
        if key in payload:
            tensor = _as_tensor(payload[key])
            if tensor is not None and tensor.ndim == 3:
                return tensor.to(torch.float32).contiguous().cpu(), key

    raise ValueError(
        "Missing residual score grid. Expected one of: "
        f"{candidate_keys}."
    )


def _smooth_score_map(score_map: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # score_map: [G, H, W]
    if kernel_size <= 1:
        return score_map
    if kernel_size % 2 == 0:
        raise ValueError(f"spatial_smooth_kernel must be odd, got {kernel_size}")
    x = score_map.unsqueeze(1)  # [G,1,H,W]
    x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return x.squeeze(1)


def _aggregate_group_scores(
    grouped: torch.Tensor,
    *,
    temporal_mean_weight: float,
    temporal_max_weight: float,
    temporal_std_weight: float,
    spatial_smooth_kernel: int,
) -> torch.Tensor:
    """
    grouped: [G, P, S] where S = H*W
    returns: [G, S]
    """
    if grouped.ndim != 3:
        raise ValueError(f"Expected grouped tensor [G,P,S], got shape {tuple(grouped.shape)}")

    G, P, S = grouped.shape
    if P <= 0 or S <= 0:
        raise ValueError(f"Invalid grouped shape: {tuple(grouped.shape)}")

    H = W = int(round(math.sqrt(S)))
    if H * W != S:
        # Fall back to pure vector aggregation when spatial shape is unknown.
        mean_score = grouped.mean(dim=1)
        max_score = grouped.max(dim=1).values
        std_score = grouped.std(dim=1, unbiased=False) if P > 1 else torch.zeros_like(mean_score)
        return (
            temporal_mean_weight * mean_score
            + temporal_max_weight * max_score
            + temporal_std_weight * std_score
        )

    grouped_map = grouped.view(G, P, H, W)
    mean_map = grouped_map.mean(dim=1)
    max_map = grouped_map.max(dim=1).values
    std_map = grouped_map.std(dim=1, unbiased=False) if P > 1 else torch.zeros_like(mean_map)

    score_map = (
        temporal_mean_weight * mean_map
        + temporal_max_weight * max_map
        + temporal_std_weight * std_map
    )
    score_map = _smooth_score_map(score_map, kernel_size=spatial_smooth_kernel)
    return score_map.reshape(G, S)


# -----------------------------
# Selection logic
# -----------------------------


def _build_anchor_groups(group_count: int, anchor_mode: str) -> set[int]:
    if group_count <= 0:
        return set()
    if anchor_mode == "none":
        return set()
    if anchor_mode == "first_group_only":
        return {0}
    raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")


def _stable_desc_argsort_1d(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x, dim=-1, descending=True, stable=True)


def _select_indices_for_one_video(
    payload: Dict[str, Any],
    *,
    delete_ratio: float,
    anchor_mode: str,
    score_key: str,
    temporal_mean_weight: float,
    temporal_max_weight: float,
    temporal_std_weight: float,
    spatial_smooth_kernel: int,
    non_anchor_min_keep_ratio: float,
) -> Dict[str, Any]:
    metadata = payload.get("metadata", {}) or {}
    llm_grid, score_key_used = _load_score_grid(payload, preferred_key=score_key)

    sampled_nframes = int(llm_grid.shape[0])
    H = int(llm_grid.shape[1])
    W = int(llm_grid.shape[2])
    tokens_per_group = H * W
    temporal_patch_size = int(metadata.get("temporal_patch_size", 2))
    temporal_pad = int(metadata.get("temporal_pad_frames", 0))
    padded_nframes = int(sampled_nframes + temporal_pad)
    if temporal_patch_size <= 0:
        raise ValueError(f"temporal_patch_size must be positive, got {temporal_patch_size}")
    group_count = max(padded_nframes // temporal_patch_size, 1)

    if temporal_pad > 0:
        pad_frames = llm_grid[-1:].repeat(temporal_pad, 1, 1)
        llm_grid = torch.cat([llm_grid, pad_frames], dim=0)

    llm_flat = llm_grid.view(padded_nframes, tokens_per_group)
    grouped = llm_flat.view(group_count, temporal_patch_size, tokens_per_group)

    # More EVS-like robust token scoring.
    group_scores = _aggregate_group_scores(
        grouped,
        temporal_mean_weight=temporal_mean_weight,
        temporal_max_weight=temporal_max_weight,
        temporal_std_weight=temporal_std_weight,
        spatial_smooth_kernel=spatial_smooth_kernel,
    )

    anchors = _build_anchor_groups(group_count=group_count, anchor_mode=anchor_mode)

    total_tokens = group_count * tokens_per_group
    target_keep = int(round(total_tokens * (1.0 - delete_ratio)))
    target_keep = max(0, min(target_keep, total_tokens))

    keep_mask = torch.zeros((group_count, tokens_per_group), dtype=torch.bool)
    for gid in anchors:
        keep_mask[gid, :] = True

    forced_keep = int(keep_mask.sum().item())
    if forced_keep > target_keep:
        # If delete_ratio is very aggressive, still preserve the anchor groups.
        target_keep = forced_keep

    non_anchor_groups = [gid for gid in range(group_count) if gid not in anchors]
    remaining_budget = target_keep - forced_keep

    # Small temporal coverage floor to reduce catastrophic starvation.
    if remaining_budget > 0 and len(non_anchor_groups) > 0 and non_anchor_min_keep_ratio > 0:
        desired_floor = int(round(tokens_per_group * non_anchor_min_keep_ratio))
        desired_floor = max(0, min(desired_floor, tokens_per_group))
        if desired_floor > 0:
            floor_per_group = min(desired_floor, remaining_budget // len(non_anchor_groups))
            if floor_per_group > 0:
                for gid in non_anchor_groups:
                    local_order = _stable_desc_argsort_1d(group_scores[gid])
                    keep_mask[gid, local_order[:floor_per_group]] = True
                remaining_budget = target_keep - int(keep_mask.sum().item())

    # Global fill with stable sorting over all remaining non-anchor tokens.
    if remaining_budget > 0 and len(non_anchor_groups) > 0:
        non_anchor_scores = group_scores[non_anchor_groups, :].clone()
        already_kept = keep_mask[non_anchor_groups, :]
        non_anchor_scores = non_anchor_scores.masked_fill(already_kept, float("-inf"))
        flat_scores = non_anchor_scores.reshape(-1)
        flat_order = _stable_desc_argsort_1d(flat_scores)
        valid = torch.isfinite(flat_scores[flat_order])
        flat_order = flat_order[valid]
        flat_order = flat_order[:remaining_budget]

        if flat_order.numel() > 0:
            local_gid = torch.div(flat_order, tokens_per_group, rounding_mode="floor")
            token_idx = torch.remainder(flat_order, tokens_per_group)
            non_anchor_gid_tensor = torch.tensor(non_anchor_groups, dtype=torch.long)
            gid_tensor = non_anchor_gid_tensor[local_gid]
            keep_mask[gid_tensor, token_idx] = True

    keep_indices: List[List[int]] = []
    for gid in range(group_count):
        idx = torch.nonzero(keep_mask[gid], as_tuple=False).view(-1).tolist()
        keep_indices.append(idx)

    actual_keep = int(keep_mask.sum().item())
    actual_delete_ratio = 1.0 - (actual_keep / max(total_tokens, 1))
    non_empty_group_count = sum(1 for x in keep_indices if len(x) > 0)
    avg_keep_non_anchor = 0.0
    if len(non_anchor_groups) > 0:
        avg_keep_non_anchor = sum(len(keep_indices[gid]) for gid in non_anchor_groups) / len(non_anchor_groups)

    return {
        "video_token_keep_indices": keep_indices,
        "tokens_per_group": tokens_per_group,
        "group_count": group_count,
        "anchor_group_count": len(anchors),
        "target_delete_ratio": float(delete_ratio),
        "actual_delete_ratio": float(actual_delete_ratio),
        "total_tokens": int(total_tokens),
        "kept_tokens": int(actual_keep),
        "non_empty_group_count": int(non_empty_group_count),
        "avg_keep_non_anchor": float(avg_keep_non_anchor),
        "score_key_used": score_key_used,
        "selector_hparams": {
            "anchor_mode": anchor_mode,
            "temporal_mean_weight": float(temporal_mean_weight),
            "temporal_max_weight": float(temporal_max_weight),
            "temporal_std_weight": float(temporal_std_weight),
            "spatial_smooth_kernel": int(spatial_smooth_kernel),
            "non_anchor_min_keep_ratio": float(non_anchor_min_keep_ratio),
        },
    }


# -----------------------------
# CLI
# -----------------------------


@dataclass
class SelectorConfig:
    input_dir: str
    output_jsonl: str
    delete_ratio: float
    anchor_mode: str
    score_key: str
    temporal_mean_weight: float
    temporal_max_weight: float
    temporal_std_weight: float
    spatial_smooth_kernel: int
    non_anchor_min_keep_ratio: float
    dataset_tsv: Optional[str] = None
    video_path_col: str = "video_path"
    index_col: str = "index"
    data_root: Optional[str] = None
    limit_files: int = 0



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Residual token selector for Qwen3-VL video pruning.")
    p.add_argument("--input_dir", type=Path, required=True, help="Directory of stage-2 .pt outputs.")
    p.add_argument("--output_jsonl", type=Path, required=True, help="Output manifest JSONL path.")
    p.add_argument("--delete_ratio", type=float, required=True, help="Target global deletion ratio in [0,1).")

    # New recommended knobs
    p.add_argument(
        "--anchor_mode",
        type=str,
        default="first_group_only",
        choices=["first_group_only", "none"],
        help="Anchor policy. Recommended: first_group_only (closer to EVS).",
    )
    p.add_argument(
        "--score_key",
        type=str,
        default="auto",
        help="Residual score key in .pt payload. Use 'auto' to prefer fp32/fp16 keys and fall back to uint8.",
    )
    p.add_argument("--temporal_mean_weight", type=float, default=0.70)
    p.add_argument("--temporal_max_weight", type=float, default=0.30)
    p.add_argument("--temporal_std_weight", type=float, default=0.00)
    p.add_argument(
        "--spatial_smooth_kernel",
        type=int,
        default=3,
        help="Odd kernel size for local averaging on score maps. Set 1 to disable.",
    )
    p.add_argument(
        "--non_anchor_min_keep_ratio",
        type=float,
        default=0.02,
        help="Small keep floor for every non-anchor group to avoid temporal starvation.",
    )

    # Dataset mapping
    p.add_argument("--dataset_tsv", type=Path, default=None, help="Optional Video-MME TSV for writing sample index.")
    p.add_argument("--video_path_col", type=str, default="video_path")
    p.add_argument("--index_col", type=str, default="index")
    p.add_argument("--data_root", type=Path, default=None, help="Optional dataset root for abs path index mapping.")
    p.add_argument("--limit_files", type=int, default=0)
    return p.parse_args()



def main() -> int:
    args = parse_args()
    cfg = SelectorConfig(
        input_dir=str(args.input_dir),
        output_jsonl=str(args.output_jsonl),
        delete_ratio=float(args.delete_ratio),
        anchor_mode=str(args.anchor_mode),
        score_key=str(args.score_key),
        temporal_mean_weight=float(args.temporal_mean_weight),
        temporal_max_weight=float(args.temporal_max_weight),
        temporal_std_weight=float(args.temporal_std_weight),
        spatial_smooth_kernel=int(args.spatial_smooth_kernel),
        non_anchor_min_keep_ratio=float(args.non_anchor_min_keep_ratio),
        dataset_tsv=None if args.dataset_tsv is None else str(args.dataset_tsv),
        video_path_col=str(args.video_path_col),
        index_col=str(args.index_col),
        data_root=None if args.data_root is None else str(args.data_root),
        limit_files=int(args.limit_files),
    )

    if not (0.0 <= cfg.delete_ratio < 1.0):
        raise ValueError(f"delete_ratio must be in [0,1), got {cfg.delete_ratio}.")
    if cfg.temporal_mean_weight < 0 or cfg.temporal_max_weight < 0 or cfg.temporal_std_weight < 0:
        raise ValueError("temporal weights must be non-negative")
    weight_sum = cfg.temporal_mean_weight + cfg.temporal_max_weight + cfg.temporal_std_weight
    if weight_sum <= 0:
        raise ValueError("At least one temporal weight must be positive")
    if not (0.0 <= cfg.non_anchor_min_keep_ratio < 1.0):
        raise ValueError(
            f"non_anchor_min_keep_ratio must be in [0,1), got {cfg.non_anchor_min_keep_ratio}"
        )

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
            anchor_mode=cfg.anchor_mode,
            score_key=cfg.score_key,
            temporal_mean_weight=cfg.temporal_mean_weight,
            temporal_max_weight=cfg.temporal_max_weight,
            temporal_std_weight=cfg.temporal_std_weight,
            spatial_smooth_kernel=cfg.spatial_smooth_kernel,
            non_anchor_min_keep_ratio=cfg.non_anchor_min_keep_ratio,
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
            "non_empty_group_count": selection["non_empty_group_count"],
            "avg_keep_non_anchor": selection["avg_keep_non_anchor"],
            "score_key_used": selection["score_key_used"],
            "selector_hparams": selection["selector_hparams"],
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
    avg_non_empty_groups = sum(r["non_empty_group_count"] for r in rows) / max(len(rows), 1)
    print(f"[DONE] wrote {len(rows)} rows to {output_jsonl}")
    print(f"[DONE] target_delete_ratio={cfg.delete_ratio:.4f}, avg_actual_delete_ratio={avg_delete_ratio:.4f}")
    print(f"[DONE] avg_non_empty_group_count={avg_non_empty_groups:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python revised_residual_token_selector.py --input_dir /path/to/stage2_pt --output_jsonl /path/to/output_keep.jsonl --delete_ratio 0.5 --anchor_mode first_group_only --score_key auto --temporal_mean_weight 0.70 --temporal_max_weight 0.30 --temporal_std_weight 0.00 --spatial_smooth_kernel 3 --non_anchor_min_keep_ratio 0.02