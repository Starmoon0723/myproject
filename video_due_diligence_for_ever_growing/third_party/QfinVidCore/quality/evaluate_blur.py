# QfinVidCore/quality/evaluate_blur.py
"""模糊度评估模块。

设计要点：
1. 通过 FFmpeg 滤镜链提取采样帧的边缘强度统计值；
2. 同时考虑全局平均与低分位弱质时段；
3. 输出 0~10 整数分，分数越高表示越模糊。
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


def _blur_index(edge_mean: float, edge_p10: float) -> float:
    # 双统计量融合：
    # - edge_mean：全局边缘强度（整体清晰度）
    # - edge_p10：低分位边缘强度（弱质时段）
    return 0.7 * edge_mean + 0.3 * edge_p10


def _blur_score_from_index(index_value: float) -> int:
    # 基于 quality_calib_blur 标定集得到的参考区间。
    sharp_ref = 54.4099
    blurry_ref = 11.8892
    if sharp_ref == blurry_ref:
        return 0

    ratio = (sharp_ref - index_value) / (sharp_ref - blurry_ref)
    ratio = max(0.0, min(1.0, ratio))

    # 非线性映射：提升中高模糊区间的区分度。
    return int(round((ratio ** 0.8) * 10.0))


def evaluate_blur(
    video_path: str | VideoEntity,
    *,
    sample_fps: Optional[float] = 1.0,
    timeout: Optional[float] = None,
) -> int:
    """评估视频模糊程度，返回 0~10 整数分（越高越模糊）。"""
    input_path = resolve_input_path(video_path)
    validate_video_input(input_path)

    effective_fps = float(sample_fps) if sample_fps is not None else 1.0
    if effective_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    # 评估滤镜链：灰度化 -> Sobel 边缘提取 -> signalstats 统计。
    vf = "format=gray,sobel=scale=2,signalstats"
    spec = build_metric_spec(input_path, vf, sample_fps=effective_fps)
    logs = run_ffmpeg_metric_or_raise("evaluate_blur", spec, timeout=timeout)

    # 从 FFmpeg 日志提取边缘强度序列（时间维度）。
    edge_values = extract_metric_values(logs, "lavfi.signalstats.YAVG")

    # 全局统计 + 低分位统计融合，兼顾稳定性与弱质片段敏感性。
    edge_mean = average(edge_values, fallback=20.0)
    edge_p10 = _percentile(edge_values, 10.0)
    index_value = _blur_index(edge_mean, edge_p10)
    return _blur_score_from_index(index_value)
