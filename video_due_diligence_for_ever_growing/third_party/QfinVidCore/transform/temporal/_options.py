# QfinVidCore/transform/temporal/_options.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from QfinVidCore.transform._runtime_options import RunOptions


@dataclass(slots=True)
class SetFpsOptions(RunOptions):
    vcodec: str = "libx264"
    crf: str = "23"
    preset: str = "veryfast"
    keep_audio: bool = True
    acodec: str = "aac"
    audio_bitrate: str = "192k"


@dataclass(slots=True)
class FadeInOutOptions(RunOptions):
    fade_duration: float = 0.6
    disable_audio: bool = False
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"
    acodec: str = "aac"
    audio_bitrate: str = "192k"
    apply_to_start: bool = True
    apply_to_end: bool = True


@dataclass(slots=True)
class ConcatenateOptions(RunOptions):
    fade_duration: float = 0.6
    fps: float | None = None
    disable_audio: bool = False
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"
    acodec: str = "aac"
    audio_bitrate: str = "192k"
    transition: bool = True
    apply_to_start: bool = True
    apply_to_end: bool = True


@dataclass(slots=True)
class SampleOptions(RunOptions):
    interval: int | None = None
    format: str | None = None
    overwrite: bool | None = None


@dataclass(slots=True)
class SplitByDurationOptions(RunOptions):
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"
    audio: Literal["copy", "aac", "drop"] = "copy"
    audio_bitrate: str = "128k"
    clean_old: bool = True


@dataclass(slots=True)
class SplitByNClipsOptions(RunOptions):
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"
    audio: Literal["copy", "aac", "drop"] = "copy"
    audio_bitrate: str = "128k"
    clean_old: bool = True


@dataclass(slots=True)
class StackFramesOptions(RunOptions):
    layout: str = "2x2"
