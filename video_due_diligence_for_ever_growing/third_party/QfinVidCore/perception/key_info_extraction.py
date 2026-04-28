# QfinVidCore/perception/key_info_extraction.py
"""
从视频中提取关键信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity


Pathlike = Union[str, Path]


def key_info_extraction(
    input_video: Union[VideoEntity, Pathlike],
    info_types: Optional[list[str]] = None,
    **kwargs
) -> Dict[str, any]:
    """
    从视频中提取关键信息。
    
    Args:
        input_video: 输入视频实体或路径
        info_types: 要提取的信息类型列表
        **kwargs: 传递给提取算法的其他参数
    
    Returns:
        Dict[str, any]: 提取的关键信息字典
    """
    # 处理输入视频
    if isinstance(input_video, VideoEntity):
        video_path = input_video.uri
        # 从VideoEntity中提取信息
        info = {
            "uri": str(input_video.uri),
            "fps": input_video.fps,
            "height": input_video.height,
            "width": input_video.width,
            "num_frames": input_video.num_frames,
            "duration_sec": input_video.duration_sec,
            "bitrate": input_video.bitrate
        }
    else:
        video_path = Path(input_video).resolve()
        # 这里可以添加代码来提取视频信息
        info = {
            "uri": str(video_path)
        }
    
    # 示例实现：添加一些默认信息
    # 实际实现中，这里应该调用具体的信息提取模型
    if info_types is None:
        info_types = ["basic", "objects", "timestamps"]
    
    if "objects" in info_types:
        info["objects"] = ["person", "car", "building"]
    
    if "timestamps" in info_types:
        info["timestamps"] = {
            "start": 0.0,
            "end": info.get("duration_sec", 0.0)
        }
    
    return info
