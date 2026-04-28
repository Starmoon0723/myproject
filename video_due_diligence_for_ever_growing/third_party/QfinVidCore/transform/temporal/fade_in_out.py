# QfinVidCore/transform/temporal/fade_in_out.py
"""时间模块：为单段视频添加淡入淡出效果。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.temporal._common import has_audio, probe_duration_seconds, resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.temporal._options import FadeInOutOptions

Pathlike = Union[str, Path]


def fade_in_out(
    input_video: Union[VideoEntity, Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    options: Optional[FadeInOutOptions] = None,
) -> Path:
    """对单段视频添加淡入/淡出效果。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - output_path: 输出文件路径；为空时自动生成。
    - options: 淡入淡出与编码运行参数集合。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: fade_duration 非法（<=0）。
    - RuntimeError: ffmpeg/ffprobe 执行失败。
    """
    opts = options or FadeInOutOptions()
    if opts.fade_duration <= 0:
        raise ValueError("fade_duration must be > 0")

    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output = input_path.parent / f"{input_path.stem}_fade{input_path.suffix}"
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    duration = probe_duration_seconds(input_path, timeout=opts.timeout)
    eps = 0.02
    d_i = min(float(opts.fade_duration), max(0.0, duration / 2.0 - eps))
    fadeout_st = max(0.0, duration - d_i)

    disable_audio = opts.disable_audio
    if not disable_audio and not has_audio(input_path, timeout=opts.timeout):
        disable_audio = True

    fadein = f"fade=t=in:st=0:d={d_i:.3f}" if opts.apply_to_start else None
    fadeout = f"fade=t=out:st={fadeout_st:.3f}:d={d_i:.3f}" if opts.apply_to_end else None

    base_v = "setpts=PTS-STARTPTS,format=yuv420p"
    if fadein and fadeout:
        v_chain = f"[0:v]{base_v},{fadein},{fadeout}[vout]"
    elif fadein:
        v_chain = f"[0:v]{base_v},{fadein}[vout]"
    elif fadeout:
        v_chain = f"[0:v]{base_v},{fadeout}[vout]"
    else:
        v_chain = f"[0:v]{base_v}[vout]"

    filters = [v_chain]
    if not disable_audio:
        afadein = f"afade=t=in:st=0:d={d_i:.3f}" if opts.apply_to_start else None
        afadeout = f"afade=t=out:st={fadeout_st:.3f}:d={d_i:.3f}" if opts.apply_to_end else None
        base_a = "asetpts=PTS-STARTPTS,aresample=async=1"

        if afadein and afadeout:
            a_chain = f"[0:a]{base_a},{afadein},{afadeout}[aout]"
        elif afadein:
            a_chain = f"[0:a]{base_a},{afadein}[aout]"
        elif afadeout:
            a_chain = f"[0:a]{base_a},{afadeout}[aout]"
        else:
            a_chain = f"[0:a]{base_a}[aout]"
        filters.append(a_chain)

    filter_complex = ";".join(filters)

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_options = {
        "filter_complex": filter_complex,
        "map": ["[vout]"],
        "c:v": opts.vcodec,
        "preset": opts.preset,
        "crf": str(opts.crf),
        "movflags": "+faststart",
    }
    if not disable_audio:
        output_options["map"] = ["[vout]", "[aout]"]
        output_options["c:a"] = opts.acodec
        output_options["b:a"] = opts.audio_bitrate
    else:
        output_options["an"] = True

    output_options.update(user_output_options)

    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output, "options": output_options}],
        "_flags": flags,
    }

    run_ffmpeg_or_raise("fade_in_out", spec, timeout=opts.timeout)
    return output
