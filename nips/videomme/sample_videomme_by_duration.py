#!/usr/bin/env python3
"""Stratified sampling for Video-MME by video duration.

Goal:
- Read full Video-MME TSV (2700 questions, 900 videos, 3 questions/video).
- Sample 1/3 videos with near-uniform coverage over video lengths.
- Keep all rows/questions for sampled videos.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def ffprobe_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    duration = float(out)
    if duration <= 0:
        raise ValueError(f"Non-positive duration: {duration}")
    return duration


def resolve_video_file(video_root: Path, rel_video_path: str) -> Optional[Path]:
    p = video_root / rel_video_path
    if p.exists():
        return p
    return None


def allocate_per_bin(bin_counts: List[int], total_target: int) -> List[int]:
    # Proportional allocation with largest-remainder correction.
    total = sum(bin_counts)
    if total == 0:
        return [0] * len(bin_counts)

    quotas = [c * total_target / total for c in bin_counts]
    floors = [min(int(math.floor(q)), c) for q, c in zip(quotas, bin_counts)]
    remaining = total_target - sum(floors)

    remainders = sorted(
        [(i, quotas[i] - floors[i]) for i in range(len(bin_counts)) if floors[i] < bin_counts[i]],
        key=lambda x: x[1],
        reverse=True,
    )
    for i, _ in remainders:
        if remaining <= 0:
            break
        floors[i] += 1
        remaining -= 1

    return floors


def build_duration_table(
    video_paths: List[str],
    video_root: Path,
    workers: int,
) -> pd.DataFrame:
    resolved: Dict[str, Path] = {}
    for vp in video_paths:
        p = resolve_video_file(video_root, vp)
        if p is None:
            raise FileNotFoundError(f"Video not found under root: {vp}")
        resolved[vp] = p

    records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2vp = {ex.submit(ffprobe_duration_seconds, path): vp for vp, path in resolved.items()}
        for fut in as_completed(fut2vp):
            vp = fut2vp[fut]
            duration = fut.result()
            records.append({"video_path": vp, "duration_sec": duration})
    return pd.DataFrame(records)


def sample_videos_stratified(duration_df: pd.DataFrame, n_bins: int, n_target_videos: int, seed: int) -> List[str]:
    duration_df = duration_df.sort_values("duration_sec").reset_index(drop=True)

    # Quantile bins. duplicates='drop' handles repeated durations.
    bins = pd.qcut(duration_df["duration_sec"], q=n_bins, labels=False, duplicates="drop")
    duration_df = duration_df.assign(bin_id=bins.astype(int))

    groups = [g for _, g in duration_df.groupby("bin_id", sort=True)]
    bin_counts = [len(g) for g in groups]
    n_per_bin = allocate_per_bin(bin_counts, n_target_videos)

    sampled = []
    for g, n_take in zip(groups, n_per_bin):
        if n_take <= 0:
            continue
        sampled_g = g.sample(n=n_take, random_state=seed)
        sampled.extend(sampled_g["video_path"].tolist())

    # Safety correction in case of edge rounding.
    if len(sampled) > n_target_videos:
        rng = np.random.default_rng(seed)
        sampled = list(rng.choice(sampled, size=n_target_videos, replace=False))
    elif len(sampled) < n_target_videos:
        remaining_pool = duration_df[~duration_df["video_path"].isin(sampled)]["video_path"].tolist()
        rng = np.random.default_rng(seed)
        add_n = min(n_target_videos - len(sampled), len(remaining_pool))
        if add_n > 0:
            sampled.extend(rng.choice(remaining_pool, size=add_n, replace=False).tolist())

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 1/3 Video-MME by duration strata.")
    parser.add_argument(
        "--input-tsv",
        type=Path,
        default=Path("/data/oceanus_ctr/j-shangshouduo-jk/myproject/data/processed/Video-MME/Video-MME.tsv"),
        help="Input TSV path.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/data"),
        help="Video root directory.",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("/data/oceanus_ctr/j-shangshouduo-jk/myproject/data/processed/Video-MME/Video-MME_900.tsv"),
        help="Output sampled TSV path.",
    )
    parser.add_argument("--fraction", type=float, default=1.0 / 3.0, help="Fraction of videos to sample.")
    parser.add_argument("--bins", type=int, default=10, help="Number of duration strata.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--workers", type=int, default=16, help="ffprobe worker threads.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    if "video_path" not in df.columns:
        raise ValueError("Input TSV must contain `video_path` column.")

    unique_videos = sorted(df["video_path"].astype(str).unique().tolist())
    n_total_videos = len(unique_videos)
    n_target_videos = max(1, int(round(n_total_videos * args.fraction)))

    duration_df = build_duration_table(unique_videos, args.video_root, args.workers)
    sampled_videos = sample_videos_stratified(duration_df, args.bins, n_target_videos, args.seed)
    sampled_set = set(sampled_videos)

    sampled_df = df[df["video_path"].astype(str).isin(sampled_set)].copy()
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df.insert(0, "index", range(len(sampled_df)))

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(args.output_tsv, sep="\t", index=False)

    # Save concise stats for reproducibility/debugging.
    stats = {
        "input_tsv": str(args.input_tsv),
        "output_tsv": str(args.output_tsv),
        "video_root": str(args.video_root),
        "seed": args.seed,
        "bins": args.bins,
        "fraction": args.fraction,
        "n_total_videos": n_total_videos,
        "n_sampled_videos": len(sampled_set),
        "n_total_rows": int(len(df)),
        "n_sampled_rows": int(len(sampled_df)),
    }
    stats_path = args.output_tsv.with_suffix(args.output_tsv.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved sampled TSV: {args.output_tsv}")
    print(f"Saved stats JSON: {stats_path}")


if __name__ == "__main__":
    main()
