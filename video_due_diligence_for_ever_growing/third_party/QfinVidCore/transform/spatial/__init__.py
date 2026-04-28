# QfinVidCore/transform/spatial/__init__.py
"""
空间转换模块（调整大小等）。
"""

from __future__ import annotations

__all__ = [
    "resize",
    "ResizeOptions",
]

from QfinVidCore.transform.spatial import resize
from QfinVidCore.transform.spatial._options import ResizeOptions
