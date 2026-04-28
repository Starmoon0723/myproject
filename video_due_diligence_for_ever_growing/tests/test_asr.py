import base64
import hashlib
import hmac
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import app
from video_due_diligence.schemas.asr import (
    ASRAckResponse,
    ASRCallbackSuccess,
    ASRResultData,
    ASRPullRequest,
    ASRPullResponse,
    ASRSubmitRequest,
    ASRTrace,
    ASRUtterance,
)

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
            hashlib.sha256,
        ).digest()
    ).decode()
    return signature


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
        "x-request-id": "test-request-id",
    }


def test_asr_submit_success():
    """测试ASR提交接口成功情况"""
    audio_data = b"test audio data"
    audio_b64 = base64.b64encode(audio_data).decode()

    submit_request = ASRSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        audio_b64=audio_b64,
        audio_format="wav",
        callback_url="http://example.com/callback",
        is_last=False,
    )

    response = client.post(
        "/api/v1/credit-av-audit/asr/submit",
        json=submit_request.dict(),
        headers=generate_headers(),
    )

    assert response.status_code == 200, response.text
    ack_response = ASRAckResponse(**response.json())
    assert ack_response.code == 0
    assert ack_response.message == "accepted"
    assert ack_response.request_id == submit_request.request_id
    assert ack_response.session_id == submit_request.session_id
    assert ack_response.accepted_at is not None


def test_asr_submit_invalid_header():
    """测试ASR提交接口缺少header情况：应返回 1001/400"""
    audio_data = b"test audio data"
    audio_b64 = base64.b64encode(audio_data).decode()

    submit_request = ASRSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        audio_b64=audio_b64,
        audio_format="wav",
        callback_url="http://example.com/callback",
        is_last=False,
    )

    response = client.post(
        "/api/v1/credit-av-audit/asr/submit",
        json=submit_request.dict(),
        headers={},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 1001


def test_asr_submit_invalid_signature():
    """测试ASR提交接口无效签名情况：应返回 1101/401"""
    audio_data = b"test audio data"
    audio_b64 = base64.b64encode(audio_data).decode()

    submit_request = ASRSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        audio_b64=audio_b64,
        audio_format="wav",
        callback_url="http://example.com/callback",
        is_last=False,
    )

    headers = generate_headers()
    headers["x-signature"] = "invalid-signature"

    response = client.post(
        "/api/v1/credit-av-audit/asr/submit",
        json=submit_request.dict(),
        headers=headers,
    )

    assert response.status_code == 401
    body = response.json()
    assert body["code"] == 1101


def test_asr_submit_invalid_audio_data():
    """测试ASR提交接口无效base64情况：应返回 1001/400"""
    submit_request = ASRSubmitRequest(
        request_id="test-request-123",
        session_id="test-session-456",
        segment_index=0,
        segment_ts_ms=0,
        audio_b64="!!!not_base64!!!",
        audio_format="wav",
        callback_url="http://example.com/callback",
        is_last=False,
    )

    response = client.post(
        "/api/v1/credit-av-audit/asr/submit",
        json=submit_request.dict(),
        headers=generate_headers(),
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 1001


def test_asr_pull_success():
    """测试ASR拉取结果接口成功情况"""
    with patch("video_due_diligence.api.asr.result_cache") as mock_cache:
        utterance = ASRUtterance(
            start_ms=0,
            end_ms=1000,
            text="test text",
            confidence=0.95,
        )

        result_data = ASRResultData(
            language="zh-CN",
            utterances=[utterance],
        )

        trace = ASRTrace(
            model="test-model",
            latency_ms=100,
        )

        callback_success = ASRCallbackSuccess(
            result_type="asr",
            request_id="test-request-123",
            session_id="test-session-456",
            segment_index=0,
            segment_ts_ms=0,
            status="SUCCESS",
            result=result_data,
            trace=trace,
        )

        mock_cache.get_result.return_value = {
            "data": callback_success.dict(),
            "trace": {
                "cached_at": "2026-02-03T16:20:10Z",
                "expired_at": "2026-02-04T16:20:10Z",
            },
        }

        pull_request = ASRPullRequest(request_id="test-request-123")
        response = client.post(
            "/api/v1/credit-av-audit/asr/pull",
            json=pull_request.dict(),
            headers=generate_headers(),
        )

        assert response.status_code == 200
        pull_response = ASRPullResponse(**response.json())
        assert pull_response.code == 0
        assert pull_response.message == "ok"
        assert pull_response.request_id == pull_request.request_id
        assert pull_response.purged is True
        assert pull_response.result is not None
        assert pull_response.result.request_id == "test-request-123"
        assert pull_response.trace is not None


def test_asr_pull_not_found():
    """测试ASR拉取结果接口未找到结果情况：应返回 1404/404"""
    with patch("video_due_diligence.api.asr.result_cache") as mock_cache:
        mock_cache.get_result.return_value = None

        pull_request = ASRPullRequest(request_id="non-existent-request")
        response = client.post(
            "/api/v1/credit-av-audit/asr/pull",
            json=pull_request.dict(),
            headers=generate_headers(),
        )

        assert response.status_code == 404
        body = response.json()
        assert body["code"] == 1404

