# QfinVidCore/transform/temporal/_common.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from QfinVidCore.transform._runtime_common import (
    as_path,
    resolve_input_path,
    run_ffmpeg_or_raise,
    run_ffprobe_json_or_raise,
    run_ffprobe_or_raise,
)


def collect_clips_from_dir(clips_dir: Path, *, exts: Sequence[str] = (".mp4", ".mov", ".mkv")) -> list[Path]:
    ext_set = {e.lower() for e in exts}
    clips = [p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_set]
    clips.sort()
    return clips


def probe_duration_seconds(path: Path, *, timeout: float | None = None) -> float:
    spec = {
        "_global": {
            "v": "error",
            "show_entries": "format=duration",
            "of": "default=noprint_wrappers=1:nokey=1",
        },
        "_inputs": [path],
    }
    out = run_ffprobe_or_raise("temporal.ffprobe_duration", spec, timeout=timeout).strip()
    return max(float(out), 0.0)


def has_audio(path: Path, *, timeout: float | None = None) -> bool:
    spec = {
        "_global": {
            "v": "error",
            "select_streams": "a:0",
            "show_entries": "stream=index",
            "of": "json",
        },
        "_inputs": [path],
    }
    data: dict[str, Any] = run_ffprobe_json_or_raise("temporal.ffprobe_audio", spec, timeout=timeout)
    return bool(data.get("streams"))


__all__ = [
    "as_path",
    "collect_clips_from_dir",
    "has_audio",
    "probe_duration_seconds",
    "resolve_input_path",
    "run_ffmpeg_or_raise",
]
