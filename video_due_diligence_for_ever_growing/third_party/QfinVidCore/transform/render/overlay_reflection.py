# QfinVidCore/transform/render/overlay_reflection.py
"""渲染模块：为视频叠加动态反光效果。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.render._common import probe_display_resolution, resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.render._options import OverlayReflectionOptions

Pathlike = Union[str, Path]


def overlay_reflection(
    input_video: Union[VideoEntity, Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    alpha: Optional[float] = None,
    options: Optional[OverlayReflectionOptions] = None,
) -> Path:
    """为视频添加动态反光层。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - output_path: 输出文件路径；为空时自动生成。
    - alpha: 反光透明度，优先级高于 options.alpha。
    - options: 渲染与运行参数集合。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: alpha 不在 [0, 1] 范围。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or OverlayReflectionOptions()
    reflection_alpha = opts.alpha if alpha is None else float(alpha)
    if not (0.0 <= reflection_alpha <= 1.0):
        raise ValueError("alpha must be in [0.0, 1.0]")

    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output = input_path.with_name(f"{input_path.stem}_reflection{input_path.suffix}")
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = probe_display_resolution(input_path, timeout=opts.timeout)

    if opts.pos:
        cx, cy = [int(x.strip()) for x in opts.pos.split(",")]
    else:
        cx, cy = width // 2, height // 2

    rx = max(1, width // 4)
    ry = max(1, width // 4)

    dist_f = f"sqrt(pow(X-{cx},2)/pow({rx},2)+pow(Y-{cy},2)/pow({ry},2))"
    alpha_f = f"if(lte({dist_f},3), 255*{reflection_alpha}*exp(-2*pow({dist_f},2)), 0)"

    nodes: list[str] = []
    nodes.append("[0:v]format=yuv420p[main]")
    nodes.append(
        f"color=c=white:s={width}x{height},format=rgba,"
        f"geq=lum='255':cb=100:cr=150:a='{alpha_f}',"
        "boxblur=25[ref_layer]"
    )
    off_x, off_y = "40*sin(1.2*t)", "25*cos(1.5*t)"
    nodes.append(f"[main][ref_layer]overlay=x='{off_x}':y='{off_y}':shortest=1:format=auto[vout]")

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

    run_ffmpeg_or_raise("overlay_reflection", spec, timeout=opts.timeout)
    return output
