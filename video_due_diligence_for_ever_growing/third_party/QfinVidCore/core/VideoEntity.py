# QfinVidCore/core/VideoEntity.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from QfinVidCore.utils.runtime import run_cmd, which

Pathlike = Union[str, Path]


@dataclass(frozen=True)
class VideoEntity:
    """
    视频实体：对齐文档中的字段设计（uri/fps/height/width/num_frames/duration_sec/bitrate）。:contentReference[oaicite:1]{index=1}
    通过 ffprobe 自动解析。
    """

    uri: Path
    fps: int
    height: int
    width: int
    num_frames: int
    duration_sec: float
    bitrate: int

    @staticmethod
    def _require_ffprobe() -> str:
        p = which("ffprobe")
        if not p:
            raise EnvironmentError("未找到 ffprobe，请先安装 ffmpeg 并保证 ffprobe 在 PATH 中。")
        return p

    @classmethod
    def from_path(cls, uri: Pathlike) -> "VideoEntity":
        uri_path = Path(uri).expanduser().resolve()
        if not uri_path.exists():
            raise FileNotFoundError(f"视频不存在：{uri_path}")

        cls._require_ffprobe()

        # 使用 ffprobe 输出 json
        args = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(uri_path),
        ]
        cp = run_cmd(args, check=True, capture_output=True, text=True)
        info = json.loads(cp.stdout)

        streams = info.get("streams", [])
        fmt = info.get("format", {})

        # 找视频流
        vstream = None
        for s in streams:
            if s.get("codec_type") == "video":
                vstream = s
                break
        if vstream is None:
            raise ValueError(f"未找到视频流：{uri_path}")

        width = int(vstream.get("width") or 0)
        height = int(vstream.get("height") or 0)

        # FPS: avg_frame_rate 形如 "30000/1001"
        fps = _parse_fraction_to_int(vstream.get("avg_frame_rate")) or _parse_fraction_to_int(
            vstream.get("r_frame_rate")
        )
        if fps is None:
            fps = 0

        duration_sec = float(fmt.get("duration") or vstream.get("duration") or 0.0)

        # num_frames：优先 nb_frames，否则用 duration * fps 近似
        nb_frames = vstream.get("nb_frames")
        if nb_frames is not None:
            try:
                num_frames = int(nb_frames)
            except Exception:
                num_frames = int(round(duration_sec * fps)) if fps > 0 else 0
        else:
            num_frames = int(round(duration_sec * fps)) if fps > 0 else 0

        # bitrate：format.bit_rate
        bit_rate = fmt.get("bit_rate") or 0
        try:
            bitrate = int(bit_rate)
        except Exception:
            bitrate = 0

        return cls(
            uri=uri_path,
            fps=int(fps),
            height=height,
            width=width,
            num_frames=num_frames,
            duration_sec=float(duration_sec),
            bitrate=bitrate,
        )

    def ensure_exists(self) -> None:
        if not self.uri.exists():
            raise FileNotFoundError(f"视频不存在：{self.uri}")


def _parse_fraction_to_int(v: Optional[str]) -> Optional[int]:
    if not v or not isinstance(v, str):
        return None
    if "/" in v:
        a, b = v.split("/", 1)
        try:
            num = float(a)
            den = float(b)
            if den == 0:
                return None
            return int(round(num / den))
        except Exception:
            return None
    try:
        return int(round(float(v)))
    except Exception:
        return None
