# QfinVidCore/quality/evaluate_shake.py
"""抖动强度评估模块。

设计目标：
1. 通过一阶/二阶帧差构建时序不稳定特征；
2. 融合两类特征得到抖动指数；
3. 输出 0~10 的整数分，分数越高表示抖动越明显。
"""

from __future__ import annotations

from typing import Optional

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.quality._common import (
    average,
    build_metric_spec,
    extract_metric_values,
    resolve_input_path,
    run_ffmpeg_metric_or_raise,
    validate_video_input,
)


def _shake_index(diff1_mean: float, diff2_mean: float) -> float:
    # 融合一阶与二阶差分：
    # - diff1 反映总体帧间变化量
    # - diff2 强调高频不稳定（更接近抖动感）
    return 0.3 * diff1_mean + 0.7 * diff2_mean


def _shake_score_from_index(index_value: float) -> int:
    # 基于 quality_calib_shake 标定集得到的参考区间。
    low_ref = 27.1952
    high_ref = 39.4065
    if high_ref == low_ref:
        return 0

    ratio = (index_value - low_ref) / (high_ref - low_ref)
    ratio = max(0.0, min(1.0, ratio))
    return int(round((ratio ** 0.9) * 10.0))


def evaluate_shake(
    video_path: str | VideoEntity,
    *,
    sample_fps: Optional[float] = 2.0,
    timeout: Optional[float] = None,
) -> int:
    """
    评估镜头抖动强度，返回 0~10 的整数分。
    分数越高表示抖动越强。
    """
    input_path = resolve_input_path(video_path)
    validate_video_input(input_path)

    effective_fps = float(sample_fps) if sample_fps is not None else 2.0
    if effective_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    # 一阶帧差：相邻帧差异，作为运动强度代理。
    spec_diff1 = build_metric_spec(
        input_path,
        "tblend=all_mode=difference,signalstats",
        sample_fps=effective_fps,
    )
    logs1 = run_ffmpeg_metric_or_raise("evaluate_shake_diff1", spec_diff1, timeout=timeout)
    diff1_values = extract_metric_values(logs1, "lavfi.signalstats.YAVG")
    diff1_mean = average(diff1_values, fallback=10.0)

    # 二阶帧差：对差分结果再差分，强调高频时序波动。
    spec_diff2 = build_metric_spec(
        input_path,
        "tblend=all_mode=difference,tblend=all_mode=difference,signalstats",
        sample_fps=effective_fps,
    )
    logs2 = run_ffmpeg_metric_or_raise("evaluate_shake_diff2", spec_diff2, timeout=timeout)
    diff2_values = extract_metric_values(logs2, "lavfi.signalstats.YAVG")
    diff2_mean = average(diff2_values, fallback=5.0)

    index_value = _shake_index(diff1_mean, diff2_mean)
    return _shake_score_from_index(index_value)
