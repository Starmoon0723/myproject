from pydantic import BaseModel
from typing import Any, List, Optional, Union


class VideoTask(BaseModel):
    """视频任务模型"""
    task_id: str
    name: str
    desc: str


class VideoGuidanceSubmitRequest(BaseModel):
    """视频尽调提交请求模型"""
    request_id: str
    session_id: str
    segment_index: int
    segment_ts_ms: int
    video_b64: str
    video_format: str
    task_list: List[VideoTask]
    callback_url: str
    risk_ruleset: str
    is_last: bool
    ext: Optional[dict]=None


class VideoGuidanceAckResponse(BaseModel):
    """视频尽调提交响应模型"""
    code: int
    message: str
    request_id: str
    session_id: str
    accepted_at: str


class HitTask(BaseModel):
    """命中的任务模型"""
    task_id: str
    video_quality: Optional[str] = None
    completation: Optional[str] = None
    scene_desc: Optional[str] = None
    risk: Optional[str] = None


class Guidance(BaseModel):
    """指导建议模型"""
    next_action: str
    suggestion: str


class VideoGuidanceResultData(BaseModel):
    """视频尽调结果数据模型"""
    hit_tasks: List[HitTask]
    guidance: Guidance


class VideoGuidanceTrace(BaseModel):
    """视频尽调处理追踪信息模型"""
    model: str
    latency_ms: int


class VideoGuidanceCallbackSuccess(BaseModel):
    """视频尽调回调成功响应模型"""
    result_type: str
    request_id: str
    session_id: str
    segment_index: int
    segment_ts_ms: int
    frames: int
    fps: int
    status: str  # SUCCESS
    # analysis: Optional[Any] = None
    result: VideoGuidanceResultData
    trace: VideoGuidanceTrace


class VideoGuidanceError(BaseModel):
    """视频尽调错误信息模型"""
    code: int
    message: str


class VideoGuidanceCallbackFailed(BaseModel):
    """视频尽调回调失败响应模型"""
    result_type: str
    request_id: str
    session_id: str
    segment_index: int
    status: str  # FAILED
    error: VideoGuidanceError


class VideoGuidancePullRequest(BaseModel):
    """视频尽调结果拉取请求模型"""
    request_id: str


class VideoGuidancePullResponse(BaseModel):
    """视频尽调结果拉取响应模型"""
    code: int
    message: str
    request_id: str
    purged: bool
    result: Optional[Union[VideoGuidanceCallbackSuccess, VideoGuidanceCallbackFailed]]
    trace: Optional[dict]