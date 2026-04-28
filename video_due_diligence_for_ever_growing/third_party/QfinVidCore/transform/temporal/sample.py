# QfinVidCore/transform/temporal/sample.py
"""时间模块：按指定帧序号抽取图片。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.core.config import get_config
from QfinVidCore.transform.temporal._common import resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.temporal._options import SampleOptions

Pathlike = Union[str, Path]


def _normalize_frame_indices(frame_indices: Iterable[int]) -> list[int]:
    """校验并标准化帧序号列表（非负、去重、排序）。"""
    normalized: list[int] = []
    for idx in frame_indices:
        i = int(idx)
        if i < 0:
            raise ValueError("frame_indices must contain non-negative integers")
        normalized.append(i)

    if not normalized:
        raise ValueError("frame_indices must not be empty")

    # 去重并排序，避免重复抽帧造成结果数量不可预期。
    return sorted(set(normalized))


def sample(
    source: Union[VideoEntity, Pathlike],
    output: Pathlike,
    frame_indices: list[int],
    *,
    options: SampleOptions | None = None,
) -> bool:
    """按显式帧序号抽帧。

    参数：
    - source: 输入视频路径或 VideoEntity。
    - output: 输出目录路径。
    - frame_indices: 要抽取的帧序号列表，例如 ``[0, 15, 30]``。
    - options: 运行参数与图片格式参数集合。

    返回：
    - bool: 抽帧文件数量与请求帧数量一致时返回 True。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: frame_indices 非法（为空或含负数）。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or SampleOptions()
    input_path = resolve_input_path(source)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    indices = _normalize_frame_indices(frame_indices)

    image_format = opts.format or get_config("transform.temporal.sample.format", "png")

    out_dir = Path(output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(out_dir / f"frame_%06d.{image_format}")

    overwrite = opts.overwrite
    if overwrite is None:
        overwrite = get_config("dependencies.ffmpeg.default_args.overwrite", True)

    # ffmpeg select：命中指定帧序号时输出一帧。
    # 例如 eq(n\,0)+eq(n\,15)+eq(n\,30)
    select_expr = "+".join([f"eq(n\\,{i})" for i in indices])

    input_options = dict(opts.input_options)
    output_options: dict[str, object] = {
        "vf": f"select='{select_expr}'",
        "vsync": "vfr",
        "frames:v": len(indices),
    }
    output_options.update(dict(opts.output_options))

    global_options = dict(opts.global_options)
    if overwrite and "y" not in global_options and "-y" not in global_options:
        global_options["y"] = True

    spec = {
        "_global": global_options,
        "_inputs": [{"path": input_path, "options": input_options}],
        "_outputs": [{"path": output_template, "options": output_options}],
        "_flags": list(opts.flags),
    }

    run_ffmpeg_or_raise("sample", spec, timeout=opts.timeout)

    output_files = sorted(out_dir.glob(f"*.{image_format}"))
    return len(output_files) == len(indices)
