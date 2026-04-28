# QfinVidCore/transform/temporal/concatenate.py
"""时间模块：拼接多段视频（可选转场）。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from QfinVidCore.transform.temporal._common import (
    as_path,
    collect_clips_from_dir,
    has_audio,
    probe_duration_seconds,
    run_ffmpeg_or_raise,
)
from QfinVidCore.transform.temporal._options import ConcatenateOptions

Pathlike = Union[str, Path]


def concatenate(
    clips: Union[Sequence[Pathlike], Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    transition: Optional[bool] = None,
    options: Optional[ConcatenateOptions] = None,
) -> Path:
    """拼接多段视频并输出成一个文件。

    参数：
    - clips: 输入片段列表，或包含片段文件的目录路径。
    - output_path: 输出路径；不传时自动命名。
    - transition: 是否启用转场，优先级高于 options.transition。
    - options: 拼接参数集合（fade 时长、编码参数、音频策略等）。

    返回：
    - Path: 生成的拼接视频路径。

    异常：
    - FileNotFoundError: 输入目录或片段文件不存在。
    - ValueError: 片段数量不足或参数非法。
    - RuntimeError: ffmpeg/ffprobe 执行失败。
    """
    opts = options or ConcatenateOptions()
    use_transition = opts.transition if transition is None else bool(transition)

    if isinstance(clips, (str, Path)):
        clips_dir = as_path(clips).resolve()
        if not clips_dir.exists():
            raise FileNotFoundError(f"clips dir not found: {clips_dir}")
        clip_paths = collect_clips_from_dir(clips_dir)
    else:
        clip_paths = [as_path(p).resolve() for p in clips]

    if len(clip_paths) < 2:
        raise ValueError("Need at least 2 clips to concatenate.")

    for p in clip_paths:
        if not p.exists():
            raise FileNotFoundError(f"clip not found: {p}")

    if output_path is None:
        output = clip_paths[0].parent / f"{clip_paths[0].stem}_merged.mp4"
    else:
        output = as_path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    durations = [probe_duration_seconds(p, timeout=opts.timeout) for p in clip_paths]
    if any(d <= 0 for d in durations):
        raise RuntimeError("ffprobe duration returned 0 for one or more clips.")

    disable_audio = opts.disable_audio
    if not disable_audio and not all(has_audio(p, timeout=opts.timeout) for p in clip_paths):
        disable_audio = True

    filters: List[str] = []
    video_labels: List[str] = []
    audio_labels: List[str] = []

    eps = 0.02
    for i, dur in enumerate(durations):
        d_i = min(float(opts.fade_duration), max(0.0, dur / 2.0 - eps))
        fadeout_st = max(0.0, dur - d_i)

        if use_transition:
            fadein = f"fade=t=in:st=0:d={d_i:.3f}" if opts.apply_to_start else None
            fadeout = f"fade=t=out:st={fadeout_st:.3f}:d={d_i:.3f}" if opts.apply_to_end else None
        else:
            fadein = None
            fadeout = None

        base_v = "setpts=PTS-STARTPTS,format=yuv420p"
        if fadein and fadeout:
            v_chain = f"[{i}:v]{base_v},{fadein},{fadeout}[vout{i}]"
        elif fadein:
            v_chain = f"[{i}:v]{base_v},{fadein}[vout{i}]"
        elif fadeout:
            v_chain = f"[{i}:v]{base_v},{fadeout}[vout{i}]"
        else:
            v_chain = f"[{i}:v]{base_v}[vout{i}]"
        filters.append(v_chain)
        video_labels.append(f"[vout{i}]")

        if not disable_audio:
            if use_transition:
                afadein = f"afade=t=in:st=0:d={d_i:.3f}" if opts.apply_to_start else None
                afadeout = f"afade=t=out:st={fadeout_st:.3f}:d={d_i:.3f}" if opts.apply_to_end else None
            else:
                afadein = None
                afadeout = None

            base_a = "asetpts=PTS-STARTPTS,aresample=async=1"
            if afadein and afadeout:
                a_chain = f"[{i}:a]{base_a},{afadein},{afadeout}[aout{i}]"
            elif afadein:
                a_chain = f"[{i}:a]{base_a},{afadein}[aout{i}]"
            elif afadeout:
                a_chain = f"[{i}:a]{base_a},{afadeout}[aout{i}]"
            else:
                a_chain = f"[{i}:a]{base_a}[aout{i}]"
            filters.append(a_chain)
            audio_labels.append(f"[aout{i}]")

    if disable_audio:
        concat_inputs = "".join(video_labels)
        filters.append(f"{concat_inputs}concat=n={len(video_labels)}:v=1:a=0[vout]")
        v_final = "[vout]"
        a_final = None
    else:
        concat_inputs = "".join([v + a for v, a in zip(video_labels, audio_labels)])
        filters.append(f"{concat_inputs}concat=n={len(video_labels)}:v=1:a=1[vout][aout]")
        v_final = "[vout]"
        a_final = "[aout]"

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_options = {
        "filter_complex": ";".join(filters),
        "map": [v_final],
        "c:v": opts.vcodec,
        "preset": opts.preset,
        "crf": str(opts.crf),
        "movflags": "+faststart",
    }
    if not disable_audio and a_final is not None:
        output_options["map"] = [v_final, a_final]
        output_options["c:a"] = opts.acodec
        output_options["b:a"] = opts.audio_bitrate
    else:
        output_options["an"] = True

    if opts.fps is not None:
        fps_value = float(opts.fps)
        if fps_value <= 0:
            raise ValueError("fps must be > 0 when provided.")
        output_options["r"] = str(fps_value)

    output_options.update(user_output_options)

    input_specs = [{"path": p, "options": input_options} for p in clip_paths]
    spec = {
        "_global": global_options,
        "_inputs": input_specs,
        "_outputs": [{"path": output, "options": output_options}],
        "_flags": flags,
    }

    run_ffmpeg_or_raise("concatenate", spec, timeout=opts.timeout)
    return output
