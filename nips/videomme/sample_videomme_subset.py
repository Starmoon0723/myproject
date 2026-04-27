#!/usr/bin/env python3
"""Sample 1/3 of Video-MME videos with duration-uniform coverage.

Default behavior for Video-MME:
- Input:  900 videos, 2700 QA rows (3 questions per video)
- Output: 300 videos, 900 QA rows

Method:
- Probe each video's duration via ffprobe.
- Sort videos by duration.
- Pick evenly spaced videos across the sorted list (systematic sampling),
  so short/medium/long ranges are all covered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def pick_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find column for {label}. Available columns: {list(df.columns)}")


def run_ffprobe_duration(video_path: Path) -> Optional[float]:
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
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return None


def resolve_video_path(video_value: str, video_root: Path) -> Path:
    p = Path(video_value)
    if p.is_absolute():
        return p
    return video_root / p


def evenly_spaced_pick(sorted_items: List[str], n_pick: int) -> List[str]:
    n_total = len(sorted_items)
    if n_pick >= n_total:
        return list(sorted_items)
    if n_pick <= 0:
        return []

    # Use centered, evenly spaced anchors to avoid edge bias.
    idxs = sorted(
        {
            min(n_total - 1, max(0, int(round((i + 0.5) * n_total / n_pick - 0.5))))
            for i in range(n_pick)
        }
    )
    # Safety: fill if rare rounding collision happened.
    if len(idxs) < n_pick:
        used = set(idxs)
        for j in range(n_total):
            if j not in used:
                idxs.append(j)
                used.add(j)
            if len(idxs) == n_pick:
                break
        idxs = sorted(idxs)
    return [sorted_items[i] for i in idxs]


def summarize_bins(durations: Dict[str, float], selected: set[str]) -> Tuple[str, str]:
    vals = pd.Series(list(durations.values()))
    q1 = vals.quantile(1 / 3)
    q2 = vals.quantile(2 / 3)

    all_counts = {"short": 0, "medium": 0, "long": 0}
    sel_counts = {"short": 0, "medium": 0, "long": 0}

    for vid, d in durations.items():
        if d <= q1:
            b = "short"
        elif d <= q2:
            b = "medium"
        else:
            b = "long"
        all_counts[b] += 1
        if vid in selected:
            sel_counts[b] += 1

    return json.dumps(all_counts, ensure_ascii=False), json.dumps(sel_counts, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample duration-uniform Video-MME subset.")
    parser.add_argument(
        "--input_tsv",
        type=Path,
        default=Path("/data/oceanus_ctr/j-shangshouduo-jk/myproject/data/processed/Video-MME/Video-MME.tsv"),
    )
    parser.add_argument(
        "--video_root",
        type=Path,
        default=Path("/data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/data"),
    )
    parser.add_argument(
        "--output_tsv",
        type=Path,
        default=Path("/data/oceanus_ctr/j-shangshouduo-jk/myproject/data/processed/Video-MME/Video-MME_900.tsv"),
    )
    parser.add_argument(
        "--target_videos",
        type=int,
        default=300,
        help="Number of videos to keep. For 1/3 of 900 videos, use 300.",
    )
    parser.add_argument(
        "--cache_json",
        type=Path,
        default=None,
        help="Optional duration cache json path. If set, probe results are reused.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    video_col = pick_column(df, ["video_path", "video", "video_id", "videoID", "video_name"], "video_path")

    unique_videos = df[video_col].astype(str).drop_duplicates().tolist()
    total_videos = len(unique_videos)
    if total_videos == 0:
        raise ValueError("No videos found in input TSV.")

    duration_cache: Dict[str, Optional[float]] = {}
    if args.cache_json and args.cache_json.exists():
        duration_cache = json.loads(args.cache_json.read_text(encoding="utf-8"))

    durations: Dict[str, float] = {}
    missing: List[str] = []
    for vid in unique_videos:
        if vid in duration_cache and duration_cache[vid] is not None:
            durations[vid] = float(duration_cache[vid])
            continue
        abs_video = resolve_video_path(vid, args.video_root)
        d = run_ffprobe_duration(abs_video)
        duration_cache[vid] = d
        if d is None:
            missing.append(vid)
        else:
            durations[vid] = d

    if args.cache_json:
        args.cache_json.parent.mkdir(parents=True, exist_ok=True)
        args.cache_json.write_text(
            json.dumps(duration_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if missing:
        print(f"[WARN] {len(missing)} videos missing duration (ffprobe failed). They will be excluded.")

    valid_videos = sorted(durations.keys(), key=lambda v: durations[v])
    if not valid_videos:
        raise ValueError("No valid video durations found. Check video paths and ffprobe.")

    target_videos = min(args.target_videos, len(valid_videos))
    picked_videos = evenly_spaced_pick(valid_videos, target_videos)
    picked_set = set(picked_videos)

    out_df = df[df[video_col].astype(str).isin(picked_set)].copy()
    if "index" in out_df.columns:
        out_df["index"] = range(len(out_df))

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_tsv, sep="\t", index=False)

    all_bins, sel_bins = summarize_bins(durations, picked_set)
    print(f"Input videos: {total_videos}, valid videos: {len(valid_videos)}")
    print(f"Picked videos: {len(picked_set)}")
    print(f"Output rows: {len(out_df)}")
    print(f"Duration bins (all): {all_bins}")
    print(f"Duration bins (picked): {sel_bins}")
    print(f"Saved to: {args.output_tsv}")


if __name__ == "__main__":
    main()
