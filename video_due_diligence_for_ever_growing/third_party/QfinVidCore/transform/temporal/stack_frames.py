# QfinVidCore/transform/temporal/stack_frames.py
"""时间模块：将视频帧按网格平铺为拼贴画面。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.temporal._common import resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.temporal._options import StackFramesOptions

Pathlike = Union[str, Path]


# 注意：rows、cols 设置过大时，输出分辨率会增大，可能触发内存不足。
def stack_frames(
    input_video: Union[VideoEntity, Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    options: Optional[StackFramesOptions] = None,
) -> Path:
    """使用 ffmpeg 的 tile 滤镜将视频帧堆叠为网格布局。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - output_path: 输出路径；不传时自动追加 `_stacked`。
    - rows/cols: 目标行列数（需同时传入），优先级高于 options.layout。
    - options: 运行参数与布局参数集合，布局格式为 `<rows>x<cols>`。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: rows/cols 参数非法或格式错误。
    - RuntimeError: ffmpeg 执行失败。
    """
    # 统一 options 入口：未传入时使用默认参数对象。
    opts = options or StackFramesOptions()

    # 行列参数必须成对出现；否则无法确定目标布局。
    if (rows is None) ^ (cols is None):
        raise ValueError("rows and cols must be provided together.")

    # 布局优先级：显式 rows/cols > options.layout。
    if rows is None and cols is None:
        try:
            resolved_rows, resolved_cols = [int(x) for x in opts.layout.split("x", 1)]
        except Exception as exc:
            raise ValueError("layout must be in '<rows>x<cols>' format, for example '2x2'") from exc
    else:
        resolved_rows = int(rows)
        resolved_cols = int(cols)

    # 行列数必须为正整数。
    if resolved_rows <= 0 or resolved_cols <= 0:
        raise ValueError("layout rows/cols must be > 0")

    # 解析输入路径并做存在性校验。
    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # 组装输出路径；未传 output_path 时自动命名。
    if output_path is None:
        output = input_path.with_stem(f"{input_path.stem}_stacked")
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 使用滑动窗口参数，避免默认 tile 使 fps 按网格数被均分。
    tile_count = resolved_rows * resolved_cols
    overlap = max(0, tile_count - 1)
    init_padding = max(0, tile_count - 1)

    input_options = dict(opts.input_options)
    output_options = {
        "vf": (
            f"tile={resolved_cols}x{resolved_rows}:"
            f"overlap={overlap}:"
            f"init_padding={init_padding}"
        ),
    }
    output_options.update(dict(opts.output_options))

    # 默认开启覆盖输出（-y），并允许用户在 global_options 中覆写。
    global_options = {"y": True, **dict(opts.global_options)}

    # 按 runner 约定的 spec 结构组装命令。
    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output, "options": output_options}],
        "_flags": list(opts.flags),
    }

    # 执行 ffmpeg，失败时抛出包含命令细节的异常。
    run_ffmpeg_or_raise("stack_frames", spec, timeout=opts.timeout)
    return output
