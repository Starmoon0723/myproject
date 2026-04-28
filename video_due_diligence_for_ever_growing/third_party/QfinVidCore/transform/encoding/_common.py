# QfinVidCore/transform/encoding/_common.py

from __future__ import annotations

from pathlib import Path
from typing import Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform._runtime_common import run_ffmpeg_or_raise as _shared_run_ffmpeg_or_raise

Pathlike = Union[str, Path]


def resolve_input_path(input_video: Union[VideoEntity, Pathlike]) -> Path:
    if isinstance(input_video, VideoEntity):
        return Path(input_video.uri).resolve()
    return Path(input_video).resolve()


def run_ffmpeg_or_raise(operation: str, spec: dict, *, timeout: float | None = None) -> None:
    _shared_run_ffmpeg_or_raise(operation, spec, timeout=timeout)
