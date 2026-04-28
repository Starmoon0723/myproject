# QfinVidCore/quality/_common.py
"""quality 子包公共工具。

职责：
1. 输入路径与格式校验；
2. 构建 FFmpeg 指标统计任务（spec）；
3. 统一执行策略（含硬件加速模式与回退）；
4. 从日志提取指标并提供通用统计函数。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.utils.runtime.ffmpeg_runner import null_sink, run_ffmpeg_spec
from QfinVidCore.utils.runtime.hwaccel import (
    detect_hwaccel_backend,
    get_hwaccel_mode,
    inject_hwaccel_spec,
)

Pathlike = Union[str, Path]
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def resolve_input_path(video: Union[VideoEntity, Pathlike]) -> Path:
    """统一输入类型，输出绝对路径。"""
    if isinstance(video, VideoEntity):
        return Path(video.uri).resolve()
    return Path(video).resolve()


def validate_video_input(video_path: Path) -> None:
    """检查输入是否存在且为受支持的视频格式。"""
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")


def _raise_failure(operation: str, prefix: str, result: Any) -> None:
    """统一异常格式，携带失败详情与可复现命令。"""
    cmd = " ".join(result.cmd)
    details = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
    raise RuntimeError(f"{operation} failed ({prefix}): {details}\ncmd: {cmd}")


def run_ffmpeg_metric_or_raise(operation: str, spec: Dict[str, Any], *, timeout: Optional[float] = None) -> str:
    """执行指标统计任务并返回日志；失败时抛异常。

    执行策略：
    - mode=off: 仅 CPU
    - mode=auto: 先硬件加速，失败回退 CPU
    - mode=force: 强制硬件路径
    """
    mode = get_hwaccel_mode()
    if mode == "off":
        result = run_ffmpeg_spec(spec, timeout=timeout)
        if result.ok:
            return f"{result.stdout}\n{result.stderr}"
        _raise_failure(operation, "cpu", result)

    backend = detect_hwaccel_backend()
    hw_spec = inject_hwaccel_spec(spec, backend)

    if mode == "force":
        if hw_spec is None:
            if backend is None:
                raise RuntimeError(f"{operation} failed (hwaccel): no supported backend detected while mode=force")
            result = run_ffmpeg_spec(spec, timeout=timeout)
            if result.ok:
                return f"{result.stdout}\n{result.stderr}"
            _raise_failure(operation, "hwaccel(force)", result)
        result = run_ffmpeg_spec(hw_spec, timeout=timeout)
        if result.ok:
            return f"{result.stdout}\n{result.stderr}"
        _raise_failure(operation, "hwaccel(force)", result)

    # auto 模式：先硬件，再 CPU 回退。
    if hw_spec is not None:
        hw_result = run_ffmpeg_spec(hw_spec, timeout=timeout)
        if hw_result.ok:
            return f"{hw_result.stdout}\n{hw_result.stderr}"

        cpu_result = run_ffmpeg_spec(spec, timeout=timeout)
        if cpu_result.ok:
            return f"{cpu_result.stdout}\n{cpu_result.stderr}"

        hw_cmd = " ".join(hw_result.cmd)
        hw_details = hw_result.stderr.strip() or hw_result.stdout.strip() or f"returncode={hw_result.returncode}"
        cpu_cmd = " ".join(cpu_result.cmd)
        cpu_details = cpu_result.stderr.strip() or cpu_result.stdout.strip() or f"returncode={cpu_result.returncode}"
        raise RuntimeError(
            f"{operation} failed (hwaccel then cpu fallback):\n"
            f"[hwaccel] {hw_details}\ncmd: {hw_cmd}\n"
            f"[cpu] {cpu_details}\ncmd: {cpu_cmd}"
        )

    result = run_ffmpeg_spec(spec, timeout=timeout)
    if result.ok:
        return f"{result.stdout}\n{result.stderr}"
    _raise_failure(operation, "cpu", result)


def build_metric_spec(video_path: Path, vf_expr: str, *, sample_fps: float) -> Dict[str, Any]:
    """构建 FFmpeg 指标统计 spec。

    说明：
    - 通过 fps=... 控制采样密度；
    - 通过 metadata=print:file=- 将帧级统计写入日志；
    - 输出到 null 设备，不落地视频文件。
    """
    return {
        "_global": {"y": True, "hide_banner": True, "loglevel": "info"},
        "_inputs": [{"path": video_path, "options": {}}],
        "_outputs": [
            {
                "path": null_sink(),
                "options": {
                    "vf": f"fps={sample_fps:.3f},{vf_expr},metadata=print:file=-",
                    "f": "null",
                    "an": True,
                },
            }
        ],
        "_flags": [],
    }


def extract_metric_values(log_text: str, metric_key: str) -> List[float]:
    """从 FFmpeg 日志中提取指定指标序列。"""
    pattern = re.compile(rf"{re.escape(metric_key)}=([-+]?\d+(?:\.\d+)?)")
    return [float(m.group(1)) for m in pattern.finditer(log_text)]


def average(values: Iterable[float], *, fallback: float) -> float:
    """求均值；序列为空时返回 fallback。"""
    items = list(values)
    if not items:
        return fallback
    return sum(items) / len(items)


def score_linear(value: float, *, bad: float, good: float, higher_is_better: bool = True) -> int:
    """线性映射到 0~10 分。

    参数：
    - bad / good: 标定区间边界
    - higher_is_better: True 表示值越高分越高
    """
    if good == bad:
        return 0

    if higher_is_better:
        ratio = (value - bad) / (good - bad)
    else:
        ratio = (bad - value) / (bad - good)

    ratio = max(0.0, min(1.0, ratio))
    return int(round(ratio * 10))
