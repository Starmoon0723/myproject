# QfinVidCore/transform/render/overlay_blur.py
"""渲染模块：对整段视频施加模糊效果。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.render._common import resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.render._options import OverlayBlurOptions

Pathlike = Union[str, Path]


def overlay_blur(
    input_video: Union[VideoEntity, Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    strength: Optional[int] = None,
    options: Optional[OverlayBlurOptions] = None,
) -> Path:
    """为视频整体添加模糊效果。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - output_path: 输出文件路径；为空时自动生成。
    - strength: 模糊强度，优先级高于 options.strength。
    - options: 渲染与运行参数集合。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: strength 非法（<=0）。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or OverlayBlurOptions()
    blur_strength = opts.strength if strength is None else int(strength)
    if blur_strength <= 0:
        raise ValueError("strength must be > 0")

    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output = input_path.with_name(f"{input_path.stem}_blur{input_path.suffix}")
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    blur_filter = (
        f"[0:v]boxblur=luma_radius={blur_strength}:luma_power=1:"
        f"chroma_radius={blur_strength}:chroma_power=1[vout]"
    )

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_options = {
        "filter_complex": blur_filter,
        "map": ["[vout]"],
        "c:v": opts.vcodec,
        "preset": opts.preset,
        "crf": str(opts.crf),
        "movflags": "+faststart",
    }
    if opts.preserve_audio:
        output_options["map"] = ["[vout]", "0:a?"]
        output_options["c:a"] = "copy"
    else:
        output_options["an"] = True
    output_options.update(user_output_options)

    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output, "options": output_options}],
        "_flags": flags,
    }

    run_ffmpeg_or_raise("overlay_blur", spec, timeout=opts.timeout)
    return output
