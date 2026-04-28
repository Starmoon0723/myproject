# QfinVidCore/transform/render/_options.py

from __future__ import annotations

from dataclasses import dataclass

from QfinVidCore.transform._runtime_options import RunOptions


@dataclass(slots=True)
class OverlayBlurOptions(RunOptions):
    strength: int = 10
    preserve_audio: bool = True
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"


@dataclass(slots=True)
class OverlayImageOptions(RunOptions):
    opacity: float = 1.0
    preserve_audio: bool = True
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"


@dataclass(slots=True)
class OverlayReflectionOptions(RunOptions):
    alpha: float = 0.5
    pos: str | None = None
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"


@dataclass(slots=True)
class OverlayShadowOptions(RunOptions):
    side: int = 1
    alpha: float = 0.8
    fade_ratio: float = 0.7
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: str = "23"
