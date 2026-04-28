# QfinVidCore/transform/encoding/to_mov.py
"""编码模块：将视频转换为 MOV 格式。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.encoding._common import resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.encoding._options import ToMovOptions

Pathlike = Union[str, Path]


def to_mov(
    input_video: Union[VideoEntity, Pathlike],
    output_path: Optional[Pathlike] = None,
    *,
    options: Optional[ToMovOptions] = None,
) -> Path:
    """将输入视频转换为 MOV 格式。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - output_path: 输出文件路径；为空时自动生成。
    - options: 编码与运行参数集合。

    返回：
    - Path: 生成的 MOV 文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or ToMovOptions()
    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        if input_path.suffix.lower() == ".mov":
            output = input_path.with_stem(f"{input_path.stem}_converted").with_suffix(".mov")
        else:
            output = input_path.with_suffix(".mov")
    else:
        output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    input_options = dict(opts.input_options)
    output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    base_output_options = {
        "c:v": opts.codec,
    }
    base_output_options.update(output_options)

    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output, "options": base_output_options}],
        "_flags": flags,
    }
    run_ffmpeg_or_raise("to_mov", spec, timeout=opts.timeout)
    return output
