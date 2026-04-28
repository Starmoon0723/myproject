from fastapi import APIRouter, Request, Depends, BackgroundTasks
from ..schemas.asr import (
    ASRSubmitRequest, ASRAckResponse, ASRPullRequest, ASRPullResponse,
    ASRCallbackSuccess, ASRResultData, ASRUtterance, ASRTrace, ASRCallbackFailed, ASRError
)
from ..utils.header_utils import HeaderHandler, DataDeserializer
from ..utils.result_cache import result_cache
from ..utils.errors import ApiError
from ..core.config import config
from datetime import datetime, timezone
import json
import time
import asyncio
import urllib.request
import urllib.error
from urllib.parse import urlparse
import io
import wave
import logging
import hashlib
import hmac
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)
if not logger.handlers:
    repo_root = Path(__file__).resolve().parents[2] 
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "callback_debug.log"
    
    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

router = APIRouter()

# 初始化工具类
api_secret = config.get("api.secret", "your-api-secret-here")
header_handler = HeaderHandler(api_secret=api_secret)
CALLBACK_TIMEOUT_SEC = 5
MAX_AUDIO_BYTES = int(config.get("api.max_audio_bytes", 10 * 1024 * 1024))  # 默认 10MB
ALLOW_EXTERNAL_CALLBACK = bool(config.get("api.allow_external_callback", False))


async def validate_request_headers(request: Request):
    """验证请求头的依赖函数"""
    return await header_handler.validate_headers(request)


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def get_audio_duration_ms(audio_bytes: bytes) -> int:
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_ms = int((frames / float(rate)) * 1000)
            return duration_ms
    except Exception as e:
        print(f"[Audio Debug] wave解析失败, 错误信息: {e}")
        # 如果解析失败（比如只有裸数据），返回一个合理的默认值或 0
        return 0

def _generate_callback_headers(secret: str, request_id: str) -> dict:
    """生成用于安全回调的鉴权请求头"""
    ts = str(int(time.time()))
    # 使用 secrets 生成 32 字符长度的 URL 安全随机字符串作为 nonce
    nonce = secrets.token_urlsafe(24) 
    
    string_to_sign = f"{secret}{ts}{nonce}"
    digest = hmac.new(secret.encode(), string_to_sign.encode(), hashlib.sha256).digest()
    sig = base64.urlsafe_b64encode(digest).decode()
    
    return {
        "x-timestamp": ts,
        "x-nonce": nonce,
        "x-signature": sig,
        "x-signature-method": "HMAC-SHA256",
        "x-signature-version": "v1",
        "x-request-id": request_id,
        "Content-Type": "application/json"
    }

def _post_json(url: str, payload: dict, headers: dict = None, timeout_sec: int = CALLBACK_TIMEOUT_SEC) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url=url,
        data=data,
        headers=req_headers,
        method="POST",
    )
    logger.info(f"准备发起回调请求 (Callback URL): {url}")
    logger.debug(f"回调 Headers: {headers}")
    logger.debug(f"回调 Payload: {json.dumps(payload, ensure_ascii=False)}")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if not (200 <= int(resp.status) < 300):
                logger.error(f"回调失败: HTTP 状态码 {resp.status}")
                raise urllib.error.HTTPError(url, resp.status, "callback failed", resp.headers, None)
            logger.info(f"回调成功: HTTP 状态码 {resp.status}")
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8') if e.fp else ""
        logger.error(f"回调遇到 HTTP 异常 (HTTPError): {e.code} - 响应体: {body}")
        raise
    except Exception as e:
        logger.error(f"回调遇到网络或其他异常: {str(e)}", exc_info=True)
        raise


def _is_local_callback_url(url: str) -> bool:
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        return host in ("127.0.0.1", "localhost")
    except Exception:
        return False


def _should_attempt_callback(url: str) -> bool:
    # 本地/单测默认禁止回调公网，避免卡住；生产可在 config.yaml 打开 allow_external_callback
    if ALLOW_EXTERNAL_CALLBACK:
        return True
    return _is_local_callback_url(url)


def _build_asr_callback_success(req: ASRSubmitRequest, asr_text: str, latency_ms: int, model_name: str, duration_ms: int) -> dict:
    utterances = [
        ASRUtterance(
            start_ms=req.segment_ts_ms,
            end_ms=req.segment_ts_ms + duration_ms,
            text=asr_text,
            confidence=0.9,
        )
    ]
    result = ASRResultData(language="zh-CN", utterances=utterances)
    trace = ASRTrace(model=model_name, latency_ms=latency_ms)
    callback = ASRCallbackSuccess(
        result_type="asr",
        request_id=req.request_id,
        session_id=req.session_id,
        segment_index=req.segment_index,
        segment_ts_ms=req.segment_ts_ms,
        status="SUCCESS",
        result=result,
        trace=trace,
    )
    return callback.dict()


def _build_asr_callback_failed(req: ASRSubmitRequest, code: int, message: str) -> dict:
    callback = ASRCallbackFailed(
        result_type="asr",
        request_id=req.request_id,
        session_id=req.session_id,
        segment_index=req.segment_index,
        status="FAILED",
        error=ASRError(code=code, message=message),
    )
    return callback.dict()


async def _process_asr_and_callback(app, req: ASRSubmitRequest, audio_b64: str, duration_ms: int) -> None:
    """
    后台任务：生成识别结果并回调；回调失败则写入缓存。
    这里用占位实现，后续你接到真实 ASR/策略能力时，把 build_* 换成真实推理即可。
    """
    try:
        analyzer = getattr(getattr(app, "state", None), "model_analyzer", None)
        t0 = time.perf_counter()
        if analyzer:
            raw = await analyzer.analyze_asr(req.request_id, audio_b64, stream=False)
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
            status = str(parsed.get("status") or "").upper()
            if status != "SUCCESS":
                raise RuntimeError(str(parsed.get("analysis") or "model error"))
            asr_text = str(parsed.get("analysis") or "")
        else:
            asr_text = "mock_asr_text"
        latency_ms = int((time.perf_counter() - t0) * 1000)
        model_name = getattr(analyzer, "asr_model", None) or "mock-asr"
        payload = _build_asr_callback_success(req, asr_text=asr_text, latency_ms=latency_ms, model_name=model_name, duration_ms=duration_ms)
    except Exception as e:
        payload = _build_asr_callback_failed(req, code=1301, message=f"internal error: {str(e)}")

    try:
        if not _should_attempt_callback(req.callback_url):
            raise RuntimeError("external callback disabled")
        callback_headers = _generate_callback_headers(api_secret, req.request_id)
        await asyncio.to_thread(_post_json, req.callback_url, payload, callback_headers)
    except Exception:
        result_cache.store_result(req.request_id, payload)


@router.post("/credit-av-audit/asr/submit", response_model=ASRAckResponse)
async def submit_asr(
    request: Request,
    asr_request: ASRSubmitRequest,
    background_tasks: BackgroundTasks,
    header_data: dict = Depends(validate_request_headers),
):
    """
    提交ASR音频分析请求
    """
    if not asr_request.callback_url:
        raise ApiError(code=1001, message="callback_url 不能为空", http_status=400)

    # format 校验（vibe.md: wav）
    if (asr_request.audio_format or "").lower() != "wav":
        raise ApiError(code=1004, message="媒体格式不支持", http_status=415)

    # 数据反序列化（用于校验合法性 + 大小限制；推理阶段按 model_analyzer 新逻辑传 b64 字符串）
    try:
        audio_data = DataDeserializer.base64_to_audio(asr_request.audio_b64)
    except Exception as e:
        raise ApiError(code=1001, message=f"base64 非法: {str(e)}", http_status=400)

    # 分段过大（vibe.md: 1003/413）
    audio_size_bytes = len(audio_data)
    if audio_size_bytes > MAX_AUDIO_BYTES:
        raise ApiError(code=1003, message="分段过大（超出服务限制）", http_status=413)

    duration_ms = get_audio_duration_ms(audio_data)
    
    # 异步处理 + 回调（失败写缓存）
    background_tasks.add_task(_process_asr_and_callback, request.app, asr_request, asr_request.audio_b64, duration_ms)
    
    # 返回ACK响应
    return ASRAckResponse(
        code=0,
        message="accepted",
        request_id=asr_request.request_id,
        session_id=asr_request.session_id,
        accepted_at=_utc_now_z()
    )


@router.post("/credit-av-audit/asr/pull", response_model=ASRPullResponse)
async def pull_asr_result(
    request: Request,
    pull_request: ASRPullRequest,
    header_data: dict = Depends(validate_request_headers)
):
    """
    拉取ASR分析结果
    """
    # 从缓存中获取结果
    result = result_cache.get_result(pull_request.request_id)
    
    if result is None:
        raise ApiError(code=1404, message="request_id 不存在或已清理", http_status=404)
    
    # 从缓存中移除已拉取的结果
    result_cache.remove_result(pull_request.request_id)
    
    # 返回结果
    return ASRPullResponse(
        code=0,
        message="ok",
        request_id=pull_request.request_id,
        purged=True,
        result=result.get("data"),
        trace=result.get("trace")
    )