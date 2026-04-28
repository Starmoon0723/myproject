# QfinVidCore/transform/render/__init__.py
"""
内容转换模块（添加图像、滤镜等）。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "overlay_image",
    "overlay_shadow",
    "overlay_reflection",
    "overlay_blur",
    "OverlayBlurOptions",
    "OverlayImageOptions",
    "OverlayReflectionOptions",
    "OverlayShadowOptions",
]

from QfinVidCore.transform.render._options import OverlayBlurOptions
from QfinVidCore.transform.render._options import OverlayImageOptions
from QfinVidCore.transform.render._options import OverlayReflectionOptions
from QfinVidCore.transform.render._options import OverlayShadowOptions


def __getattr__(name: str) -> Any:
    if name in {"overlay_image", "overlay_shadow", "overlay_reflection", "overlay_blur"}:
        module = import_module(f"QfinVidCore.transform.render.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
