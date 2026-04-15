#!/usr/bin/env python3
"""
Build offline codec cache for videos.

This script precomputes per-video codec-domain data once:
- ffprobe frame metadata
- PyAV motion vectors (with optional fallback)

The output cache is designed to be consumed by:
`codec_frame_sampler_without_extract_mvs.py --offline_cache_dir ...`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from codec_frame_sampler_without_extract_mvs import (
    av,
    cache_json_path_for_video,
    ffprobe_frames,
    find_videos,
    parse_motion_vectors_with_pyav,
    save_offline_video_data,
)


def process_one_video_to_cache(
    video_path: Path,
    input_dir: Path,
    cache_dir: Path,
    allow_pktsize_fallback: bool,
    skip_existing: bool,
) -> Tuple[Path, bool, Dict[str, Any]]:
    cache_path = cache_json_path_for_video(cache_dir, input_dir, video_path)
    if skip_existing and cache_path.is_file():
        return video_path, True, {"status": "skipped_existing", "cache_path": str(cache_path)}

    frames = ffprobe_frames(video_path)
    mv_used = False
    frame_to_mvs = None
    warning = None

    try:
        frame_to_mvs = parse_motion_vectors_with_pyav(video_path)
        mv_used = True
    except Exception as exc:
        if not allow_pktsize_fallback:
            raise
        warning = f"PyAV motion-vector read failed; fallback to packet-size-only mode: {exc}"
        frame_to_mvs = None

    meta = {
        "video_path": str(video_path),
        "num_frames": len(frames),
        "num_gops": max([f.gop_id for f in frames], default=-1) + 1 if frames else 0,
        "motion_extractor_used": mv_used,
        "warning": warning,
    }
    save_offline_video_data(
        out_path=cache_path,
        input_dir=input_dir,
        video_path=video_path,
        frames=frames,
        frame_to_mvs=frame_to_mvs,
        meta=meta,
    )
    return video_path, False, {"status": "ok", "cache_path": str(cache_path), "meta": meta}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline codec cache for later fast frame sampling runs")
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(os.cpu_count() or 4, 4))
    p.add_argument("--allow_pktsize_fallback", action="store_true", help="Allow packet-size-only fallback when PyAV MV read fails")
    p.add_argument("--limit_videos", type=int, default=0, help="Only process first N videos for quick checks")
    p.add_argument("--skip_existing", action="store_true", help="Skip videos that already have cache files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if args.limit_videos and args.limit_videos > 0:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"No mp4 videos found under: {input_dir}", file=sys.stderr)
        return 1

    if av is None and not args.allow_pktsize_fallback:
        print(
            "PyAV is not available. Please install it with `pip install av`, "
            "or pass --allow_pktsize_fallback to build packet-size-only cache.",
            file=sys.stderr,
        )
        return 2

    print(f"[INFO] Found {len(videos)} videos")
    print(f"[INFO] Cache dir: {cache_dir}")
    print(f"[INFO] Motion extractor: {'pyav' if av is not None else 'NONE (packet-size-only fallback)'}")

    results: List[Dict[str, Any]] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                process_one_video_to_cache,
                video,
                input_dir,
                cache_dir,
                args.allow_pktsize_fallback,
                args.skip_existing,
            ): video
            for video in videos
        }
        for fut in as_completed(futs):
            video = futs[fut]
            try:
                video_path, skipped, info = fut.result()
                results.append(
                    {
                        "video_path": str(video_path),
                        "skipped": skipped,
                        **info,
                    }
                )
                if skipped:
                    print(f"[SKIP] {video}")
                else:
                    print(f"[OK] cached {video}")
            except Exception as exc:
                failed += 1
                print(f"[ERR] failed {video}: {exc}", file=sys.stderr)

    summary = {
        "input_dir": str(input_dir),
        "cache_dir": str(cache_dir),
        "workers": args.workers,
        "allow_pktsize_fallback": args.allow_pktsize_fallback,
        "total_videos": len(videos),
        "failed": failed,
        "succeeded_or_skipped": len(videos) - failed,
        "results": results,
    }
    (cache_dir / "cache_build_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if failed > 0:
        print(f"[DONE] completed with failures: failed={failed} / total={len(videos)}", file=sys.stderr)
        return 3
    print(f"[DONE] cache ready: total={len(videos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
