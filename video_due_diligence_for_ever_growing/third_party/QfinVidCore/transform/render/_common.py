# QfinVidCore/transform/render/_common.py

from __future__ import annotations

from pathlib import Path

from QfinVidCore.transform._runtime_common import (
    resolve_input_path,
    run_ffmpeg_or_raise,
    run_ffprobe_json_or_raise,
)


def probe_display_resolution(path: Path, *, timeout: float | None = None) -> tuple[int, int]:
    spec = {
        "_global": {
            "v": "error",
            "select_streams": "v:0",
            "show_entries": "stream=width,height:stream_tags=rotate",
            "of": "json",
        },
        "_inputs": [path],
    }
    data = run_ffprobe_json_or_raise("render.ffprobe_resolution", spec, timeout=timeout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video streams found for: {path}")

    s0 = streams[0]
    width = int(s0["width"])
    height = int(s0["height"])
    rotate = str((s0.get("tags") or {}).get("rotate", ""))
    if rotate in {"90", "270", "-90"}:
        return height, width
    return width, height


__all__ = [
    "probe_display_resolution",
    "resolve_input_path",
    "run_ffmpeg_or_raise",
]
