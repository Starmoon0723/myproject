# QfinVidCore/transform/render/overlay_image.py
"""渲染模块：将图片以覆盖层方式叠加到视频。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.render._common import probe_display_resolution, resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.render._options import OverlayImageOptions

Pathlike = Union[str, Path]


def overlay_image(
    input_video: Union[VideoEntity, Pathlike],
    image_path: Pathlike,
    output_path: Optional[Pathlike] = None,
    *,
    options: Optional[OverlayImageOptions] = None,
) -> Path:
    """将图片水印按“等比包含 + 居中”方式叠加到视频。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - image_path: 水印图片路径。
    - output_path: 输出文件路径；为空时自动生成。
    - options: 渲染与运行参数集合。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频或水印图片不存在。
    - ValueError: opacity 不在 [0, 1] 范围。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or OverlayImageOptions()
    if not (0.0 <= opts.opacity <= 1.0):
        raise ValueError("opacity must be in [0.0, 1.0]")

    input_path = resolve_input_path(input_video)
    watermark_path = Path(image_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not watermark_path.exists():
        raise FileNotFoundError(f"Watermark image not found: {watermark_path}")

    width, height = probe_display_resolution(input_path, timeout=opts.timeout)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    with Image.open(watermark_path).convert("RGBA") as wm:
        wm_w, wm_h = wm.size
        scale = min(width / wm_w, height / wm_h)
        new_size = (max(1, int(wm_w * scale)), max(1, int(wm_h * scale)))
        wm_resized = wm.resize(new_size, Image.Resampling.LANCZOS)

    ox = (width - new_size[0]) // 2
    oy = (height - new_size[1]) // 2
    canvas.paste(wm_resized, (ox, oy), wm_resized)

    if output_path is None:
        output = input_path.with_stem(f"{input_path.stem}_wm_contain")
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    temp_wm_path = output.parent / f"temp_wm_layer_{os.getpid()}.png"
    canvas.save(temp_wm_path)

    filter_complex = f"[1:v]colorchannelmixer=aa={opts.opacity}[wm];[0:v][wm]overlay=0:0"

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_options = {
        "filter_complex": filter_complex,
        "c:v": opts.vcodec,
        "preset": opts.preset,
        "crf": str(opts.crf),
        "movflags": "+faststart",
    }
    if opts.preserve_audio:
        output_options["c:a"] = "copy"
    else:
        output_options["an"] = True
    output_options.update(user_output_options)

    spec = {
        "_global": global_options,
        "_inputs": [
            {"path": input_path, "options": input_options},
            temp_wm_path,
        ],
        "_outputs": [{"path": output, "options": output_options}],
        "_flags": flags,
    }

    try:
        run_ffmpeg_or_raise("overlay_image", spec, timeout=opts.timeout)
    finally:
        if temp_wm_path.exists():
            temp_wm_path.unlink()

    return output
