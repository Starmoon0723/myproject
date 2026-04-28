# QfinVidCore/transform/spatial/resize.py
"""空间模块：调整视频分辨率。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.core.config import get_config
from QfinVidCore.transform.spatial._common import resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.spatial._options import ResizeOptions

Pathlike = Union[str, Path]


def resize(
    input_video: Union[VideoEntity, Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    options: Optional[ResizeOptions] = None,
) -> Path:
    """将输入视频缩放到目标宽高并输出新文件。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - output_path: 输出路径；不传时自动命名。
    - width: 目标宽度，优先级高于 options 与配置文件。
    - height: 目标高度，优先级高于 options 与配置文件。
    - options: 运行参数集合（覆盖策略、ffmpeg 透传等）。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: width/height 非法（<=0）。
    - RuntimeError: ffmpeg 执行失败。
    """
    # 统一 options 入口：未传入时使用默认参数对象。
    opts = options or ResizeOptions()
    # 解析输入路径并做存在性校验。
    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # 宽高优先级：显式参数 > options > 全局配置 > 硬编码默认值。
    resolved_width = (
        width
        if width is not None
        else (opts.width if opts.width is not None else get_config("transform.spatial.resize.width", 1280))
    )
    resolved_height = (
        height
        if height is not None
        else (opts.height if opts.height is not None else get_config("transform.spatial.resize.height", 720))
    )

    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("width and height must be > 0")

    # 组装输出路径；未传 output_path 时自动命名。
    if output_path is None:
        output = input_path.with_stem(f"{input_path.stem}_{resolved_width}x{resolved_height}")
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 覆盖策略优先读取 options，缺省时回退到全局配置。
    overwrite = opts.overwrite
    if overwrite is None:
        overwrite = get_config("dependencies.ffmpeg.default_args.overwrite", True)

    # 拆分 ffmpeg 参数：输入参数、输出参数、全局参数、附加 flags。
    input_options = dict(opts.input_options)
    output_options = dict(opts.output_options)
    global_options = dict(opts.global_options)
    flags = list(opts.flags)

    # 在允许覆盖且未显式给出时，自动补 -y。
    if overwrite and "y" not in global_options and "-y" not in global_options:
        global_options["y"] = True

    # 核心缩放参数通过 vf=scale=WxH 注入。
    base_output_options = {
        "vf": f"scale={resolved_width}:{resolved_height}",
    }
    base_output_options.update(output_options)

    # 按 runner 约定的 spec 结构组装命令。
    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output, "options": base_output_options}],
        "_flags": flags,
    }

    # 执行 ffmpeg，失败时抛出包含命令细节的异常。
    run_ffmpeg_or_raise("resize", spec, timeout=opts.timeout)
    return output
