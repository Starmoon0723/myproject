# QfinVidCore/transform/encoding/set_bitrate.py
"""编码模块：按目标码率重编码视频（支持一遍/两遍编码）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.transform.encoding._common import resolve_input_path, run_ffmpeg_or_raise
from QfinVidCore.transform.encoding._options import SetBitrateOptions
from QfinVidCore.utils.runtime.ffmpeg_runner import null_sink

Pathlike = Union[str, Path]


# 文件名安全化：将不适合文件名的字符统一替换为 `_`。
def _sanitize_for_filename(s: str) -> str:
    """将任意字符串转换为安全文件名片段。"""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def set_bitrate(
    input_video: Union[VideoEntity, Pathlike],
    bitrate: str,
    output_path: Optional[Pathlike] = None,
    *,
    options: Optional[SetBitrateOptions] = None,
) -> Path:
    """按目标码率重编码视频。

    参数：
    - input_video: 输入视频路径或 VideoEntity。
    - bitrate: 目标视频码率，例如 ``"3M"``、``"1200k"``。
    - output_path: 输出文件路径；为空时自动生成。
    - options: 高级编码参数（两遍编码、音频策略、VBV、ffmpeg 透传等）。

    返回：
    - Path: 生成的视频文件路径。

    异常：
    - FileNotFoundError: 输入视频不存在。
    - ValueError: 音频模式不受支持等参数错误。
    - RuntimeError: ffmpeg 执行失败。
    """
    opts = options or SetBitrateOptions()
    input_path = resolve_input_path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        safe_bitrate = _sanitize_for_filename(bitrate)
        output_stem = f"{input_path.stem}_bitrate_{safe_bitrate}"
        output_path = input_path.with_stem(output_stem)
    else:
        output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    passlog_dir_value = opts.passlog_dir if opts.passlog_dir is not None else output_path.parent
    passlog_dir = Path(passlog_dir_value).resolve()
    passlog_dir.mkdir(parents=True, exist_ok=True)
    passlog_prefix = passlog_dir / f"ffmpeg2pass_{output_path.stem}"

    in_opts: Dict[str, Any] = dict(opts.input_options)
    user_out_opts: Dict[str, Any] = dict(opts.output_options)
    glob_opts: Dict[str, Any] = {"y": True, **dict(opts.global_options)}
    flags = list(opts.flags)

    audio_options: Dict[str, Any]
    if opts.audio == "copy":
        audio_options = {"c:a": "copy"}
    elif opts.audio == "aac":
        audio_options = {"c:a": "aac", "b:a": opts.audio_bitrate}
    else:
        raise ValueError(f"Unsupported audio mode: {opts.audio!r}")

    vbv_options: Dict[str, Any] = {}
    if opts.use_vbv:
        vbv_options["maxrate"] = opts.maxrate or bitrate
        vbv_options["bufsize"] = opts.bufsize or "2M"

    if opts.two_pass:
        pass1_spec = {
            "_global": glob_opts,
            "_inputs": [{"path": input_path, "options": in_opts}],
            "_outputs": [
                {
                    "path": null_sink(),
                    "options": {
                        "c:v": opts.codec,
                        "b:v": bitrate,
                        "pass": "1",
                        "passlogfile": str(passlog_prefix),
                        "an": True,
                        **vbv_options,
                        "f": "mp4",
                    },
                }
            ],
            "_flags": flags,
        }
        pass2_spec = {
            "_global": glob_opts,
            "_inputs": [{"path": input_path, "options": in_opts}],
            "_outputs": [
                {
                    "path": output_path,
                    "options": {
                        "c:v": opts.codec,
                        "b:v": bitrate,
                        "pass": "2",
                        "passlogfile": str(passlog_prefix),
                        **vbv_options,
                        **audio_options,
                        "movflags": "+faststart",
                        **user_out_opts,
                    },
                }
            ],
            "_flags": flags,
        }

        run_ffmpeg_or_raise("set_bitrate", pass1_spec, timeout=opts.timeout)
        run_ffmpeg_or_raise("set_bitrate", pass2_spec, timeout=opts.timeout)

        if opts.cleanup_passlog:
            for p in passlog_dir.glob(passlog_prefix.name + "*"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

        return output_path

    one_pass_spec = {
        "_global": glob_opts,
        "_inputs": [{"path": input_path, "options": in_opts}],
        "_outputs": [
            {
                "path": output_path,
                "options": {
                    "c:v": opts.codec,
                    "b:v": bitrate,
                    **vbv_options,
                    **audio_options,
                    "movflags": "+faststart",
                    **user_out_opts,
                },
            }
        ],
        "_flags": flags,
    }
    run_ffmpeg_or_raise("set_bitrate", one_pass_spec, timeout=opts.timeout)
    return output_path
