# QfinVidCore/quality/evaluate_brightness.py
"""亮度评估模块。

设计目标：
1. 基于 FFmpeg 的 signalstats 提取采样帧亮度统计；
2. 同时建模全局亮度与低分位暗部风险；
3. 输出 0~10 的整数分，分数越高表示越亮。
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


def _percentile(values: list[float], p: float) -> float:
    """计算百分位（线性插值）。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _brightness_index(y_mean: float, y_p10: float) -> float:
    # 双统计量融合：
    # - y_mean：全局平均亮度
    # - y_p10：低分位亮度（暗部/弱质时段）
    return 0.7 * y_mean + 0.3 * y_p10


def _brightness_score_from_index(index_value: float) -> int:
    # 基于 quality_calib_brightness 标定集得到的参考区间。
    dark_ref = 39.6208
    bright_ref = 396.4257
    if bright_ref == dark_ref:
        return 0

    ratio = (index_value - dark_ref) / (bright_ref - dark_ref)
    ratio = max(0.0, min(1.0, ratio))

    # 轻度非线性映射：提升暗端区间的区分度。
    return int(round((ratio ** 0.9) * 10.0))


def evaluate_brightness(
    video_path: str | VideoEntity,
    *,
    sample_fps: Optional[float] = 1.0,
    timeout: Optional[float] = None,
) -> int:
    """
    评估视频亮度，返回 0~10 的整数分。
    分数越高表示视频越亮。
    """
    input_path = resolve_input_path(video_path)
    validate_video_input(input_path)

    effective_fps = float(sample_fps) if sample_fps is not None else 1.0
    if effective_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    # 使用 signalstats 直接提取帧级亮度统计。
    spec = build_metric_spec(input_path, "signalstats", sample_fps=effective_fps)
    logs = run_ffmpeg_metric_or_raise("evaluate_brightness", spec, timeout=timeout)
    # 提取亮度序列（时间维度）。
    y_values = extract_metric_values(logs, "lavfi.signalstats.YAVG")

    # 全局亮度 + 低分位暗部融合，避免平均值掩盖暗时段。
    y_mean = average(y_values, fallback=120.0)
    y_p10 = _percentile(y_values, 10.0)
    index_value = _brightness_index(y_mean, y_p10)
    return _brightness_score_from_index(index_value)
