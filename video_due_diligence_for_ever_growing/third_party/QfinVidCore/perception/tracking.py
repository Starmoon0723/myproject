# QfinVidCore/perception/tracking.py
"""
视频中对象跟踪功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, Dict

from QfinVidCore.core.VideoEntity import VideoEntity


Pathlike = Union[str, Path]


class TrackedObject:
    """
    被跟踪的对象类。
    """
    def __init__(self, object_id: int, object_type: str, confidence: float):
        self.object_id = object_id
        self.object_type = object_type
        self.confidence = confidence
        self.bboxes = []  # 边界框列表，每个元素为(x, y, width, height, timestamp)
    
    def add_bbox(self, x: float, y: float, width: float, height: float, timestamp: float):
        """
        添加对象的边界框。
        """
        self.bboxes.append((x, y, width, height, timestamp))
    
    def to_dict(self) -> Dict:
        """
        将对象转换为字典。
        """
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "bboxes": self.bboxes
        }


def tracking(
    input_video: Union[VideoEntity, Pathlike],
    object_types: Optional[List[str]] = None,
    **kwargs
) -> List[TrackedObject]:
    """
    在视频中跟踪对象。
    
    Args:
        input_video: 输入视频实体或路径
        object_types: 要跟踪的对象类型列表
        **kwargs: 传递给跟踪模型的其他参数
    
    Returns:
        List[TrackedObject]: 跟踪到的对象列表
    """
    # 处理输入视频
    if isinstance(input_video, VideoEntity):
        video_path = input_video.uri
    else:
        video_path = Path(input_video).resolve()
    
    # 示例实现：返回一个默认跟踪对象
    # 实际实现中，这里应该调用具体的对象跟踪模型
    tracked_objects = []
    
    # 创建一个示例跟踪对象
    obj = TrackedObject(
        object_id=1,
        object_type="person",
        confidence=0.9
    )
    
    # 添加一些示例边界框
    obj.add_bbox(100, 100, 200, 300, 0.0)
    obj.add_bbox(105, 105, 200, 300, 1.0)
    obj.add_bbox(110, 110, 200, 300, 2.0)
    
    tracked_objects.append(obj)
    
    return tracked_objects
