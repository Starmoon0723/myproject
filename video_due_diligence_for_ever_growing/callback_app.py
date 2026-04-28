from fastapi import FastAPI, HTTPException
from typing import Any, Dict, Optional
import threading
import time
import uvicorn


app = FastAPI(
    title="Mock Callback Service",
    description="用于本地/测试环境接收回调的最小服务",
    version="1.0.0",
)


_lock = threading.Lock()
_asr_payload_by_request_id: Dict[str, Dict[str, Any]] = {}
_video_payload_by_request_id: Dict[str, Dict[str, Any]] = {}


def _store(store_dict: Dict[str, Dict[str, Any]], payload: Dict[str, Any]) -> str:
    request_id = str(payload.get("request_id") or f"unknown-{int(time.time() * 1000)}")
    with _lock:
        store_dict[request_id] = payload
    return request_id


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/callback/asr")
async def asr_callback(payload: Dict[str, Any]):
    request_id = _store(_asr_payload_by_request_id, payload)
    return {"code": 0, "message": "received", "request_id": request_id}


@app.get("/callback/asr/{request_id}")
async def get_asr_callback(request_id: str):
    with _lock:
        payload = _asr_payload_by_request_id.get(request_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="not found")
    return payload


@app.post("/callback/video-guidance")
async def video_guidance_callback(payload: Dict[str, Any]):
    request_id = _store(_video_payload_by_request_id, payload)
    return {"code": 0, "message": "received", "request_id": request_id}


@app.get("/callback/video-guidance/{request_id}")
async def get_video_guidance_callback(request_id: str):
    with _lock:
        payload = _video_payload_by_request_id.get(request_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="not found")
    return payload

if __name__ == "__main__":
    print("启动本地 Mock 回调服务...")
    print("稍后请将发送请求的 CALLBACK_URL 改为: http://127.0.0.1:8001/callback/video-guidance")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001
    )