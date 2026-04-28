# QfinVidCore/transform/temporal/__init__.py
"""
时间相关的视频转换模块。
"""

from __future__ import annotations

__all__ = [
    "split_by_duration",
    "split_by_n_clips",
    "concatenate",
    "fade_in_out",
    "set_fps",
    "sample",
    "stack_frames",
    "ConcatenateOptions",
    "FadeInOutOptions",
    "SampleOptions",
    "SetFpsOptions",
    "SplitByDurationOptions",
    "SplitByNClipsOptions",
    "StackFramesOptions",
]

from QfinVidCore.transform.temporal import split_by_duration
from QfinVidCore.transform.temporal import split_by_n_clips
from QfinVidCore.transform.temporal import concatenate
from QfinVidCore.transform.temporal import fade_in_out
from QfinVidCore.transform.temporal import set_fps
from QfinVidCore.transform.temporal import sample
from QfinVidCore.transform.temporal import stack_frames
from QfinVidCore.transform.temporal._options import ConcatenateOptions
from QfinVidCore.transform.temporal._options import FadeInOutOptions
from QfinVidCore.transform.temporal._options import SampleOptions
from QfinVidCore.transform.temporal._options import SetFpsOptions
from QfinVidCore.transform.temporal._options import SplitByDurationOptions
from QfinVidCore.transform.temporal._options import SplitByNClipsOptions
from QfinVidCore.transform.temporal._options import StackFramesOptions
