# QfinVidCore/transform/render/overlay_shadow.py
"""渲染模块：在视频边缘叠加阴影效果。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.render._common import probe_display_resolution, resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.render._options import OverlayShadowOptions

Pathlike = Union[str, Path]


def overlay_shadow(
    input_video: Union[VideoEntity, Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    side: Optional[int] = None,
    alpha: Optional[float] = None,
    fade_ratio: Optional[float] = None,
    options: Optional[OverlayShadowOptions] = None,
) -> Path:
    """为视频边缘添加阴影效果。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - output_path: 输出文件路径；为空时自动生成。
    - side: 阴影方向（0/1/2/3），优先级高于 options.side。
    - alpha: 阴影透明度，优先级高于 options.alpha。
    - fade_ratio: 渐变比例，优先级高于 options.fade_ratio。
    - options: 渲染与运行参数集合。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: side/alpha/fade_ratio 参数非法。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or OverlayShadowOptions()
    shadow_side = opts.side if side is None else int(side)
    shadow_alpha = opts.alpha if alpha is None else float(alpha)
    shadow_fade_ratio = opts.fade_ratio if fade_ratio is None else float(fade_ratio)

    if shadow_side not in {0, 1, 2, 3}:
        raise ValueError("side must be one of {0, 1, 2, 3}")
    if not (0.0 <= shadow_alpha <= 1.0):
        raise ValueError("alpha must be in [0.0, 1.0]")
    if not (0.0 < shadow_fade_ratio <= 1.0):
        raise ValueError("fade_ratio must be in (0.0, 1.0]")

    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output = input_path.with_name(f"{input_path.stem}_shadow{input_path.suffix}")
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = probe_display_resolution(input_path, timeout=opts.timeout)
    fade_w = max(1, int((width if shadow_side in {0, 1} else height) * shadow_fade_ratio))

    if shadow_side == 0:
        grad = f"clip(1-(X+Y*0.1)/{fade_w},0,1)"
    elif shadow_side == 1:
        grad = f"clip((X+Y*0.1-({width}-{fade_w}))/{fade_w},0,1)"
    elif shadow_side == 2:
        grad = f"clip(1-(Y+X*0.1)/{fade_w},0,1)"
    else:
        grad = f"clip((Y+X*0.1-({height}-{fade_w}))/{fade_w},0,1)"

    smooth_grad = f"({grad}*{grad}*(3-2*{grad}))"
    s_alpha_f = f"255*{shadow_alpha}*{smooth_grad}"

    nodes: list[str] = []
    nodes.append("[0:v]format=yuv420p[main]")
    nodes.append(
        f"color=c=black:s={width}x{height},format=rgba,"
        f"geq=lum=0:cb=128:cr=128:a='{s_alpha_f}',"
        "boxblur=5[shd_layer]"
    )
    nodes.append("[main][shd_layer]overlay=shortest=1:format=auto[vout]")

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_options = {
        "filter_complex": ";".join(nodes),
        "map": ["[vout]", "0:a?"],
        "c:v": opts.vcodec,
        "preset": opts.preset,
        "crf": str(opts.crf),
        "c:a": "copy",
        "movflags": "+faststart",
    }
    output_options.update(user_output_options)

    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output, "options": output_options}],
        "_flags": flags,
    }

    run_ffmpeg_or_raise("overlay_shadow", spec, timeout=opts.timeout)
    return output
