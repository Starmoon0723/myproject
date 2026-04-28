# QfinVidCore/transform/temporal/set_fps.py
"""时间模块：重采样视频帧率。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.temporal._common import resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.temporal._options import SetFpsOptions

Pathlike = Union[str, Path]


def _fps_tag(fps: float) -> str:
    """将 fps 值转换为适合文件名的标签。"""
    if abs(fps - int(fps)) < 1e-6:
        return str(int(fps))
    return f"{fps:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _default_output_path(input_path: Path, fps: float) -> Path:
    """生成默认输出路径。"""
    return input_path.with_name(f"{input_path.stem}_fps{_fps_tag(fps)}{input_path.suffix}")


def set_fps(
    input_video: Union[VideoEntity, Pathlike],
    fps: float,
    output_path: Optional[Pathlike] = None,
    *,
    options: Optional[SetFpsOptions] = None,
) -> Path:
    """将视频重采样到目标帧率（CFR）并输出。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - fps: 目标帧率（>0）。
    - output_path: 输出文件路径；为空时自动生成。
    - options: 编码与运行参数集合。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: fps 非法（<=0）。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or SetFpsOptions()
    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    fps_value = float(fps)
    if fps_value <= 0:
        raise ValueError("fps must be > 0")

    output = Path(output_path).resolve() if output_path else _default_output_path(input_path, fps_value)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_options = {
        "fps_mode": "cfr",
        "r": f"{fps_value:.3f}",
        "c:v": opts.vcodec,
        "crf": str(opts.crf),
        "preset": opts.preset,
    }
    if opts.keep_audio:
        output_options.update({"c:a": opts.acodec, "b:a": opts.audio_bitrate})
    else:
        output_options["an"] = True
    output_options.update(user_output_options)

    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output, "options": output_options}],
        "_flags": flags,
    }

    run_ffmpeg_or_raise("set_fps", spec, timeout=opts.timeout)
    return output
