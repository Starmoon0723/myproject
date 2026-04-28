# QfinVidCore/transform/temporal/split_by_n_clips.py
"""时间模块：按目标片段数切分视频。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.temporal._common import probe_duration_seconds, resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.temporal._options import SplitByNClipsOptions

Pathlike = Union[str, Path]


def split_by_n_clips(
    input_video: Union[VideoEntity, Pathlike],
    n: int,
    output_dir: Optional[Pathlike] = None,
    *,
    ensure_evenly: bool = False,
    options: Optional[SplitByNClipsOptions] = None,
) -> List[Path]:
    """将视频切成 n 段，可控制是否尽量等长。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - n: 目标片段数（>0）。
    - output_dir: 输出目录；为空时使用输入视频所在目录。
    - ensure_evenly: 是否按近似等长切分全部片段。
    - options: 编码与运行参数集合。

    返回：
    - List[Path]: 生成的分段文件路径列表。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: n 非法（<=0）或参数错误。
    - RuntimeError: ffmpeg/ffprobe 执行失败。
    """
    opts = options or SplitByNClipsOptions()
    if n <= 0:
        raise ValueError("n must be > 0")

    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    out_dir = input_path.parent if output_dir is None else Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if opts.clean_old:
        for f in out_dir.glob(f"{input_path.stem}_part_*{input_path.suffix}"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    total = probe_duration_seconds(input_path, timeout=opts.timeout)
    if total <= 0:
        return []

    seg = total / float(n)

    if opts.audio == "copy":
        audio_options: Dict[str, Any] = {"c:a": "copy"}
    elif opts.audio == "aac":
        audio_options = {"c:a": "aac", "b:a": opts.audio_bitrate}
    elif opts.audio == "drop":
        audio_options = {"an": True}
    else:
        raise ValueError(f"Unsupported audio mode: {opts.audio!r}")

    input_options = dict(opts.input_options)
    user_output_options = dict(opts.output_options)
    global_options = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    output_files: List[Path] = []
    start = 0.0
    for i in range(n):
        # 分段策略：
        # - ensure_evenly=False：前 n-1 段固定为 seg，最后一段用余量兜底。
        # - ensure_evenly=True：所有分段都按 seg，尽量保持等长。
        seg_len = seg if (ensure_evenly or i < n - 1) else max(0.0, total - start)
        if seg_len <= 0.0001:
            break

        out_path = out_dir / f"{input_path.stem}_part_{i:03d}{input_path.suffix}"
        output_options: Dict[str, Any] = {
            "ss": f"{start:.3f}",
            "t": f"{seg_len:.3f}",
            "c:v": opts.vcodec,
            "preset": opts.preset,
            "crf": str(opts.crf),
            **audio_options,
            "movflags": "+faststart",
        }
        output_options.update(user_output_options)

        spec = {
            "_global": global_options,
            "_inputs": [{"path": input_path, "options": input_options}],
            "_outputs": [{"path": out_path, "options": output_options}],
            "_flags": flags,
        }

        run_ffmpeg_or_raise("split_by_n_clips", spec, timeout=opts.timeout)
        output_files.append(out_path)
        start += seg

    return output_files
