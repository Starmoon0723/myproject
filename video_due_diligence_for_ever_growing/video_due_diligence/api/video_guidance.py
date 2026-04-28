from fastapi import APIRouter, Request, Depends, BackgroundTasks, Response
from ..schemas.video_guidance import (
    VideoGuidanceSubmitRequest, VideoGuidanceAckResponse,
    VideoGuidancePullRequest, VideoGuidancePullResponse,
    VideoGuidanceCallbackSuccess, VideoGuidanceResultData, HitTask, Guidance, VideoGuidanceTrace,
    VideoGuidanceCallbackFailed, VideoGuidanceError
)
from ..utils.header_utils import HeaderHandler, DataDeserializer
from ..utils.result_cache import result_cache
from ..utils.errors import ApiError
from ..core.config import config
from datetime import datetime, timezone
import json
import time
import asyncio
import base64
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error
from urllib.parse import urlparse
import tempfile
import logging
import hashlib
import hmac
import secrets


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
MAX_VIDEO_BYTES = int(config.get("api.max_video_bytes", 80 * 1024 * 1024))
ALLOW_EXTERNAL_CALLBACK = bool(config.get("api.allow_external_callback", False))


async def validate_request_headers(request: Request):
    """验证请求头的依赖函数"""
    return await header_handler.validate_headers(request)


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
    if ALLOW_EXTERNAL_CALLBACK:
        return True
    return _is_local_callback_url(url)


def _video_preprocess_cfg() -> dict:
    return {
        "use_video_path": bool(config.get("video_preprocess.use_video_path", False)),
        "order": list(config.get("video_preprocess.order", ["set_fps", "resize"]) or []),
        "tmp_dir": str(config.get("video_preprocess.tmp_dir", "") or ""),
        "fps": float(config.get("video_preprocess.fps", 0) or 0),
        "width": int(config.get("video_preprocess.width", 0) or 0),
        "height": int(config.get("video_preprocess.height", 0) or 0),
    }


def _is_existing_file_path(value: str) -> bool:
    try:
        return isinstance(value, str) and value.strip() != "" and os.path.exists(value)
    except Exception:
        return False


def _ensure_qfinvidcore_on_sys_path() -> None:
    """
    让 third_party/QfinVidCore 以顶层模块名 QfinVidCore 可导入。
    QfinVidCore 源码内部使用的是 `import QfinVidCore...`，因此必须把 third_party 加到 sys.path。
    """
    repo_root = Path(__file__).resolve().parents[2]
    third_party_dir = repo_root / "third_party"
    sp = str(third_party_dir)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _preprocess_video_path_to_b64(video_path: str) -> tuple[str, int, int]:
    """
    输入：本地视频路径
    输出：处理后的 base64 字符串（不带 data: 前缀）、frames(-1 暂不计算)、fps(回调展示用)

    处理：按 config.yaml 的 video_preprocess.order 依次执行 set_fps / resize。
    """
    cfg = _video_preprocess_cfg()
    in_path = Path(video_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"video path not found: {in_path}")

    _ensure_qfinvidcore_on_sys_path()
    from QfinVidCore.transform.temporal.set_fps import set_fps as qf_set_fps  # type: ignore
    from QfinVidCore.transform.spatial.resize import resize as qf_resize  # type: ignore

    tmp_dir = cfg["tmp_dir"]
    if tmp_dir:
        out_dir = Path(tmp_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.gettempdir()).resolve()

    current = in_path
    applied_fps: int = -1

    for step in cfg["order"]:
        step = str(step).strip().lower()
        if step == "set_fps":
            fps_val = float(cfg["fps"])
            if fps_val > 0:
                applied_fps = int(round(fps_val))
                out_path = out_dir / f"{current.stem}_fps{applied_fps}{current.suffix}"
                current = qf_set_fps(str(current), fps=fps_val, output_path=str(out_path))
        elif step == "resize":
            w = int(cfg["width"])
            h = int(cfg["height"])
            if w > 0 and h > 0:
                out_path = out_dir / f"{current.stem}_{w}x{h}{current.suffix}"
                current = qf_resize(str(current), output_path=str(out_path), width=w, height=h)

    b64 = base64.b64encode(Path(current).read_bytes()).decode()
    # frames 暂不计算（如需可接入 cv2/ffprobe）
    return b64, -1, applied_fps


def _build_video_guidance_callback_success(
    req: VideoGuidanceSubmitRequest,
    analysis: object,
    latency_ms: int,
    model_name: str,
    frames: int,
    fps: int,
) -> dict:
    # analysis_jsonable = analysis
    # if analysis is not None and not isinstance(analysis, (dict, list, str, int, float, bool)):
    #     analysis_jsonable = str(analysis)

    answer_map = analysis

    hit_tasks = []
    for t in req.task_list:
        task_result = answer_map.get(t.name)
        if not task_result or not isinstance(task_result, dict):
            continue
        detail = str(task_result.get("详情", "")).strip()
        quality = str(task_result.get("清晰度", "未知")).strip()
        completeness = str(task_result.get("完整性", "未知")).strip()
        risk = str(task_result.get("风险", "")).strip() # 如果模型支持风险字段
        if not detail:
            continue
        hit_tasks.append(
            HitTask(
                task_id=t.task_id,
                video_quality=quality,
                completation=completeness,
                scene_desc=detail,
                risk=risk,
            )
        )

    # suggestion：
    # - 若命中：汇总命中的任务项证据
    # - 若未命中：给出通用采集建议
    if hit_tasks:
        # suggestion = "; ".join([ht.scene_desc for ht in hit_tasks if ht.scene_desc])
        suggestion = ""
    else:
        # suggestion = "未命中待处理任务项，请保持画面稳定并对准目标拍摄 3 秒以上"
        suggestion = ""

    guidance = Guidance(
        next_action="NEXT",
        suggestion=suggestion[:500] if len(suggestion) > 500 else suggestion,
    )
    result = VideoGuidanceResultData(hit_tasks=hit_tasks, guidance=guidance)
    trace = VideoGuidanceTrace(model=model_name, latency_ms=latency_ms)
    callback = VideoGuidanceCallbackSuccess(
        result_type="video_guidance",
        request_id=req.request_id,
        session_id=req.session_id,
        segment_index=req.segment_index,
        segment_ts_ms=req.segment_ts_ms,
        frames=frames,
        fps=fps,
        status="SUCCESS",
        # analysis=analysis_jsonable,
        result=result,
        trace=trace,
    )
    return callback.dict()


def _build_video_guidance_callback_failed(req: VideoGuidanceSubmitRequest, code: int, message: str) -> dict:
    callback = VideoGuidanceCallbackFailed(
        result_type="video_guidance",
        request_id=req.request_id,
        session_id=req.session_id,
        segment_index=req.segment_index,
        status="FAILED",
        error=VideoGuidanceError(code=code, message=message),
    )
    return callback.dict()


async def _process_video_guidance_and_callback(app, req: VideoGuidanceSubmitRequest, video_b64: str, frames: int, fps: int) -> None:
    try:
        analyzer = getattr(getattr(app, "state", None), "model_analyzer", None)
        t0 = time.perf_counter()
        if analyzer:
            cfg = _video_preprocess_cfg()
            if cfg["use_video_path"] and _is_existing_file_path(video_b64):
                processed_b64, processed_frames, processed_fps = _preprocess_video_path_to_b64(video_b64)
                video_b64 = processed_b64
                if frames == -1:
                    frames = processed_frames
                if fps == -1:
                    fps = processed_fps
            task_names = [t.name for t in req.task_list]
            raw = await analyzer.analyze_video(req.request_id, video_b64, task_list=task_names, stream=False)
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
            status = str(parsed.get("status") or "").upper()
            if status != "SUCCESS":
                raise RuntimeError(str(parsed.get("analysis") or "model error"))
            analysis = parsed.get("analysis")
        else:
            analysis = {"summary": "mock_video_summary"}
        latency_ms = int((time.perf_counter() - t0) * 1000)
        model_name = getattr(analyzer, "video_model", None) or "mock-video"
        payload = _build_video_guidance_callback_success(req, analysis=analysis, latency_ms=latency_ms, model_name=model_name, frames=frames, fps=fps)
    except Exception as e:
        payload = _build_video_guidance_callback_failed(req, code=1301, message=f"internal error: {str(e)}")
    try:
        if not _should_attempt_callback(req.callback_url):
            raise RuntimeError("external callback disabled")
        
        callback_headers = _generate_callback_headers(api_secret, req.request_id)

        await asyncio.to_thread(_post_json, req.callback_url, payload, callback_headers)
    except Exception:
        result_cache.store_result(req.request_id, payload)


@router.post("/credit-av-audit/video-guidance/submit", response_model=VideoGuidanceAckResponse)
async def submit_video_guidance(
    request: Request,
    video_request: VideoGuidanceSubmitRequest,
    background_tasks: BackgroundTasks,
    header_data: dict = Depends(validate_request_headers),
):
    """
    提交视频尽调分析请求
    """
    if not video_request.callback_url:
        raise ApiError(code=1001, message="callback_url 不能为空", http_status=400)
    if not video_request.task_list:
        raise ApiError(code=1001, message="task_list 不能为空", http_status=400)

    if (video_request.video_format or "").lower() != "mp4":
        raise ApiError(code=1004, message="媒体格式不支持", http_status=415)

    cfg = _video_preprocess_cfg()
    frames, fps = -1, -1

    # 兼容：video_b64 字段既可能是 base64，也可能是本地路径（当 use_video_path=true）
    if cfg["use_video_path"] and _is_existing_file_path(video_request.video_b64):
        # 路径模式：校验文件大小，不做 base64 解码
        video_path = Path(video_request.video_b64).resolve()
        try:
            size = int(video_path.stat().st_size)
        except Exception as e:
            raise ApiError(code=1001, message=f"video 路径不可读: {str(e)}", http_status=400)
        if size > MAX_VIDEO_BYTES:
            raise ApiError(code=1003, message="分段过大（超出服务限制）", http_status=413)
        # 回调里展示预期目标 fps（实际处理成功后会在后台任务里覆盖/确认）
        fps_cfg = int(round(float(cfg["fps"]))) if float(cfg["fps"]) > 0 else -1
        fps = fps_cfg
        background_tasks.add_task(_process_video_guidance_and_callback, request.app, video_request, video_request.video_b64, frames, fps)
    else:
        # base64 模式：严格校验合法性 + 大小限制
        try:
            video_data = DataDeserializer.base64_to_video(video_request.video_b64)
        except Exception as e:
            raise ApiError(code=1001, message=f"base64 非法: {str(e)}", http_status=400)
        if len(video_data) > MAX_VIDEO_BYTES:
            raise ApiError(code=1003, message="分段过大（超出服务限制）", http_status=413)
        background_tasks.add_task(_process_video_guidance_and_callback, request.app, video_request, video_request.video_b64, frames, fps)
    
    # 返回ACK响应
    return VideoGuidanceAckResponse(
        code=0,
        message="accepted",
        request_id=video_request.request_id,
        session_id=video_request.session_id,
        accepted_at=_utc_now_z()
    )


@router.post("/credit-av-audit/video-guidance/pull", response_model=VideoGuidancePullResponse)
async def pull_video_guidance_result(
    request: Request,
    pull_request: VideoGuidancePullRequest,
    header_data: dict = Depends(validate_request_headers)
):
    """
    拉取视频尽调分析结果
    """
    # 从缓存中获取结果
    result = result_cache.get_result(pull_request.request_id)
    
    if result is None:
        raise ApiError(code=1404, message="request_id 不存在或已清理", http_status=404)
        # return Response(status_code=204, headers={"Retry-After": str(PULL_RETRY_AFTER_SEC)})

    # 有 key 但还没有产出结果：返回 204（轮询继续）
    if result.get("data") is None:
        return Response(status_code=204, headers={"Retry-After": str(PULL_RETRY_AFTER_SEC)})
        # return VideoGuidancePullResponse(
        #     code=204,
        #     message="no content",
        #     request_id=pull_request.request_id,
        #     purged=False,
        #     result=None,
        #     trace=None
        # )
    
    # 仅在成功返回结果（200）后，才从缓存中移除
    result_cache.remove_result(pull_request.request_id)
    
    # 返回结果
    return VideoGuidancePullResponse(
        code=0,
        message="ok",
        request_id=pull_request.request_id,
        purged=True,
        result=result.get("data"),
        trace=result.get("trace")
    )