import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app
from video_due_diligence.schemas.video_guidance import (
    VideoGuidanceSubmitRequest, VideoGuidanceAckResponse,
    VideoGuidancePullRequest, VideoGuidancePullResponse,
    VideoGuidanceCallbackSuccess, VideoGuidanceResultData,
    HitTask, Guidance, VideoGuidanceTrace, VideoTask
)
import base64
import json
import time
import hmac
import hashlib

client = TestClient(app)

# 测试用的API密钥
TEST_API_SECRET = "your-api-secret-here"

# 生成测试用的签名
def generate_signature(timestamp, nonce):
    string_to_sign = f"{TEST_API_SECRET}{timestamp}{nonce}"
    signature = base64.urlsafe_b64encode(
        hmac.new(
            TEST_API_SECRET.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).digest()
    ).decode()
    return signature

# 生成测试用的headers
def generate_headers():
    timestamp = str(int(time.time()))
    nonce = "test-nonce"
    signature = generate_signature(timestamp, nonce)
    
    return {
        "x-timestamp": timestamp,
        "x-nonce": nonce,
        "x-signature": signature,
        "x-signature-method": "HMAC-SHA256",
        "x-signature-version": "v1",
        "x-request-id": "test-request-id"
    }

def test_video_guidance_submit_success():
    """测试视频尽调提交接口成功情况"""
    # 准备测试数据
    video_data = b"test video data"
    video_b64 = base64.b64encode(video_data).decode()
    
    task = VideoTask(
        task_id="task-1",
        name="test task",
        desc="test task description"
    )
    
    submit_request = VideoGuidanceSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        video_b64=video_b64,
        video_format="mp4",
        task_list=[task],
        callback_url="http://example.com/callback",
        risk_ruleset="default",
        is_last=False,
        ext={"key": "value"}
    )
    
    # 发送请求
    response = client.post(
        "/api/v1/credit-av-audit/video-guidance/submit",
        json=submit_request.dict(),
        headers=generate_headers()
    )
    
    # 验证响应
    assert response.status_code == 200
    ack_response = VideoGuidanceAckResponse(**response.json())
    assert ack_response.code == 0
    assert ack_response.message == "accepted"
    assert ack_response.request_id == submit_request.request_id
    assert ack_response.session_id == submit_request.session_id
    assert ack_response.accepted_at is not None

def test_video_guidance_submit_invalid_header():
    """测试视频尽调提交接口无效header情况"""
    # 准备测试数据
    video_data = b"test video data"
    video_b64 = base64.b64encode(video_data).decode()
    
    task = VideoTask(
        task_id="task-1",
        name="test task",
        desc="test task description"
    )
    
    submit_request = VideoGuidanceSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        video_b64=video_b64,
        video_format="mp4",
        task_list=[task],
        callback_url="http://example.com/callback",
        risk_ruleset="default",
        is_last=False,
        ext={"key": "value"}
    )
    
    # 发送请求（缺少必要header）
    response = client.post(
        "/api/v1/credit-av-audit/video-guidance/submit",
        json=submit_request.dict(),
        headers={}  # 缺少header
    )
    
    # 验证响应
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 1001

def test_video_guidance_submit_invalid_signature():
    """测试视频尽调提交接口无效签名情况"""
    # 准备测试数据
    video_data = b"test video data"
    video_b64 = base64.b64encode(video_data).decode()
    
    task = VideoTask(
        task_id="task-1",
        name="test task",
        desc="test task description"
    )
    
    submit_request = VideoGuidanceSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        video_b64=video_b64,
        video_format="mp4",
        task_list=[task],
        callback_url="http://example.com/callback",
        risk_ruleset="default",
        is_last=False,
        ext={"key": "value"}
    )
    
    # 生成无效签名
    headers = generate_headers()
    headers["x-signature"] = "invalid-signature"
    
    # 发送请求
    response = client.post(
        "/api/v1/credit-av-audit/video-guidance/submit",
        json=submit_request.dict(),
        headers=headers
    )
    
    # 验证响应
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 1101

def test_video_guidance_submit_invalid_video_data():
    """测试视频尽调提交接口无效视频数据情况"""
    # 准备测试数据（无效的base64数据）
    task = VideoTask(
        task_id="task-1",
        name="test task",
        desc="test task description"
    )
    
    submit_request = VideoGuidanceSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        video_b64="!!!not_base64!!!",
        video_format="mp4",
        task_list=[task],
        callback_url="http://example.com/callback",
        risk_ruleset="default",
        is_last=False,
        ext={"key": "value"}
    )
    
    # 发送请求
    response = client.post(
        "/api/v1/credit-av-audit/video-guidance/submit",
        json=submit_request.dict(),
        headers=generate_headers()
    )
    
    # 验证响应
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 1001

def test_video_guidance_pull_success():
    """测试视频尽调拉取结果接口成功情况"""
    # 模拟缓存中有数据
    with patch("video_due_diligence.api.video_guidance.result_cache") as mock_cache:
        # 准备测试数据
        hit_task = HitTask(
            task_id="task-1",
            video_quality="good",
            completation="complete",
            scene_desc="test scene",
            risk="low"
        )
        
        guidance = Guidance(
            next_action="next",
            suggestion="suggestion"
        )
        
        result_data = VideoGuidanceResultData(
            hit_tasks=[hit_task],
            guidance=guidance
        )
        
        trace = VideoGuidanceTrace(
            model="test-model",
            latency_ms=100
        )
        
        callback_success = VideoGuidanceCallbackSuccess(
            result_type="video_guidance",
            request_id="test-request-123",
            session_id="test-session-456",
            segment_index=0,
            segment_ts_ms=0,
            frames=30,
            fps=30,
            status="SUCCESS",
            result=result_data,
            trace=trace
        )
        
        mock_cache.get_result.return_value = {
            "data": callback_success.dict(),
            "trace": trace.dict()
        }
        
        # 发送请求
        pull_request = VideoGuidancePullRequest(request_id="test-request-123")
        response = client.post(
            "/api/v1/credit-av-audit/video-guidance/pull",
            json=pull_request.dict(),
            headers=generate_headers()
        )
        
        # 验证响应
        assert response.status_code == 200
        pull_response = VideoGuidancePullResponse(**response.json())
        assert pull_response.code == 0
        assert pull_response.message == "ok"
        assert pull_response.request_id == pull_request.request_id
        assert pull_response.purged == True
        assert pull_response.result is not None
        assert pull_response.result.request_id == "test-request-123"

def test_video_guidance_pull_not_found():
    """测试视频尽调拉取结果接口未找到结果情况"""
    # 模拟缓存中没有数据
    with patch("video_due_diligence.api.video_guidance.result_cache") as mock_cache:
        mock_cache.get_result.return_value = None
        
        # 发送请求
        pull_request = VideoGuidancePullRequest(request_id="non-existent-request")
        response = client.post(
            "/api/v1/credit-av-audit/video-guidance/pull",
            json=pull_request.dict(),
            headers=generate_headers()
        )
        
        # 验证响应
        assert response.status_code == 404
        body = response.json()
        assert body["code"] == 1404