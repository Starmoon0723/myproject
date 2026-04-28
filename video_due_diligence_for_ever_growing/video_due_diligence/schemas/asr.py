from pydantic import BaseModel
from typing import List, Optional, Union


class ASRSubmitRequest(BaseModel):
    """ASR提交请求模型"""
    request_id: str
    session_id: str
    segment_index: int
    segment_ts_ms: int
    audio_b64: str
    audio_format: str
    callback_url: str
    is_last: bool


class ASRAckResponse(BaseModel):
    """ASR提交响应模型"""
    code: int
    message: str
    request_id: str
    session_id: str
    accepted_at: str


class ASRUtterance(BaseModel):
    """ASR识别结果中的语句模型"""
    start_ms: int
    end_ms: int
    text: str
    confidence: float


class ASRResultData(BaseModel):
    """ASR识别结果数据模型"""
    language: str
    utterances: List[ASRUtterance]


class ASRTrace(BaseModel):
    """ASR处理追踪信息模型"""
    model: str
    latency_ms: int


class ASRCallbackSuccess(BaseModel):
    """ASR回调成功响应模型"""
    result_type: str
    request_id: str
    session_id: str
    segment_index: int
    segment_ts_ms: int
    status: str  # SUCCESS, PARTIAL
    result: ASRResultData
    trace: ASRTrace


class ASRError(BaseModel):
    """ASR错误信息模型"""
    code: int
    message: str


class ASRCallbackFailed(BaseModel):
    """ASR回调失败响应模型"""
    result_type: str
    request_id: str
    session_id: str
    segment_index: int
    status: str  # FAILED
    error: ASRError


class ASRPullRequest(BaseModel):
    """ASR结果拉取请求模型"""
    request_id: str


class ASRPullResponse(BaseModel):
    """ASR结果拉取响应模型"""
    code: int
    message: str
    request_id: str
    purged: bool
    result: Optional[Union[ASRCallbackSuccess, ASRCallbackFailed]]
    trace: Optional[dict]