# QfinVidCore/transform/temporal/split_by_duration.py
"""时间模块：按固定时长切分视频。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.temporal._common import probe_duration_seconds, resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.temporal._options import SplitByDurationOptions

Pathlike = Union[str, Path]


def split_by_duration(
    input_video: Union[VideoEntity, Pathlike],
    duration: float,
    output_dir: Optional[Pathlike] = None,
    *,
    ensure_evenly: bool = False,
    drop_last: bool = False,
    options: Optional[SplitByDurationOptions] = None,
) -> List[Path]:
    """按时长切分视频，可控制是否均分与是否丢弃尾段。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - duration: 目标分段时长（秒）。
    - output_dir: 输出目录；为空时使用输入视频所在目录。
    - ensure_evenly: 是否按总时长均分每段时长。
    - drop_last: 是否丢弃最后不足时长的尾段。
    - options: 编码与运行参数集合。

    返回：
    - List[Path]: 生成的分段文件路径列表。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: duration 非法（<=0）或参数错误。
    - RuntimeError: ffmpeg/ffprobe 执行失败。
    """
    opts = options or SplitByDurationOptions()
    if duration <= 0:
        raise ValueError("duration must be > 0")

    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    out_dir = input_path.parent if output_dir is None else Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if opts.clean_old:
        for f in out_dir.glob(f"{input_path.stem}_part_*{input_path.suffix}"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    total = probe_duration_seconds(input_path, timeout=opts.timeout)
    if total <= 0:
        return []

    # 分段策略：
    # - ensure_evenly=False：以 duration 为固定分段长度；最后一段可能不足 duration。
    # - drop_last=True：丢弃最后不足 duration 的尾段。
    # - ensure_evenly=True：将总时长均分为 n 段，尽量保证每段时长一致。
    if ensure_evenly:
        n_parts = int(math.floor(total / duration)) if drop_last else int(math.ceil(total / duration))
        if n_parts <= 0:
            return []
        segment_duration = total / float(n_parts)
    else:
        n_parts = int(math.floor(total / duration)) if drop_last else int(math.ceil(total / duration))
        if n_parts <= 0:
            return []
        segment_duration = float(duration)

    if opts.audio == "copy":
        audio_options: Dict[str, Any] = {"c:a": "copy"}
    elif opts.audio == "aac":
        audio_options = {"c:a": "aac", "b:a": opts.audio_bitrate}
    elif opts.audio == "drop":
        audio_options = {"an": True}
    else:
        raise ValueError(f"Unsupported audio mode: {opts.audio!r}")

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_files: List[Path] = []
    for i in range(n_parts):
        start = i * segment_duration
        if ensure_evenly:
            # 均分模式下每段长度尽量相同，最后一段兜底避免浮点累积误差。
            seg_len = max(0.0, total - start) if i == n_parts - 1 else segment_duration
        else:
            seg_len = min(segment_duration, total - start)
        if seg_len <= 0:
            break

        out_path = out_dir / f"{input_path.stem}_part_{i:03d}{input_path.suffix}"
        output_options: Dict[str, Any] = {
            "ss": f"{start:.3f}",
            "t": f"{seg_len:.3f}",
            "c:v": opts.vcodec,
            "preset": opts.preset,
            "crf": str(opts.crf),
            **audio_options,
            "movflags": "+faststart",
        }
        output_options.update(user_output_options)

        spec = {
            "_global": global_options,
            "_inputs": [{"path": input_path, "options": input_options}],
            "_outputs": [{"path": out_path, "options": output_options}],
            "_flags": flags,
        }
        run_ffmpeg_or_raise("split_by_duration", spec, timeout=opts.timeout)
        output_files.append(out_path)

    return output_files
