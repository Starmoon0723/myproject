# QfinVidCore/transform/encoding/_options.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

Pathlike = Union[str, Path]


@dataclass(slots=True)
class RunOptions:
    """Shared ffmpeg execution options."""

    timeout: Optional[float] = None
    input_options: Mapping[str, Any] = field(default_factory=dict)
    output_options: Mapping[str, Any] = field(default_factory=dict)
    global_options: Mapping[str, Any] = field(default_factory=dict)
    flags: Sequence[Any] = field(default_factory=tuple)


@dataclass(slots=True)
class SetBitrateOptions(RunOptions):
    """Domain options for set_bitrate."""

    two_pass: bool = True
    codec: str = "libx264"
    audio: Literal["copy", "aac"] = "copy"
    audio_bitrate: str = "128k"
    passlog_dir: Optional[Pathlike] = None
    cleanup_passlog: bool = True
    use_vbv: bool = False
    bufsize: Optional[str] = None
    maxrate: Optional[str] = None


@dataclass(slots=True)
class ToMovOptions(RunOptions):
    """Domain options for to_mov."""

    codec: str = "prores"


@dataclass(slots=True)
class ToMp4Options(RunOptions):
    """Domain options for to_mp4."""

    codec: Optional[str] = None
    overwrite: Optional[bool] = None
