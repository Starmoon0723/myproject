"""
客户端版端到端测试脚本（不负责启动服务）：

- 你需要自己先启动：
  1) 主服务：python -m uvicorn video_due_diligence.main:app --host 127.0.0.1 --port 8000
  2) 回调服务：python -m uvicorn callback_app:app --host 127.0.0.1 --port 8001

- 然后运行本脚本：
    python scripts\\e2e_client_pipeline.py

用途：
  submit -> 等待 callback_app 收到回调（超时则尝试 /pull）
"""

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


MAIN_BASE = os.environ.get("E2E_MAIN_BASE", "http://127.0.0.1:8000")
CALLBACK_BASE = os.environ.get("E2E_CALLBACK_BASE", "http://127.0.0.1:8001")
WAIT_SEC = int(os.environ.get("E2E_WAIT_SEC", "60"))
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_AUDIO_PATH = DATA_DIR / "test.wav"
DEFAULT_VIDEO_PATH = DATA_DIR / "IMG_2663_000.mp4"


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} {url}: {body}") from e


def _load_api_secret() -> str:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    try:
        import yaml  # type: ignore
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return ((cfg.get("api") or {}).get("secret")) or "your-api-secret-here"
    except Exception:
        return "your-api-secret-here"


def _read_file_b64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到测试文件：{path}")
    return base64.b64encode(path.read_bytes()).decode()


def _generate_signature(api_secret: str, timestamp: str, nonce: str) -> str:
    string_to_sign = f"{api_secret}{timestamp}{nonce}"
    digest = hmac.new(api_secret.encode(), string_to_sign.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode()


def _generate_headers(api_secret: str, request_id: str) -> Dict[str, str]:
    ts = str(int(time.time()))
    nonce = "e2e-nonce"
    sig = _generate_signature(api_secret, ts, nonce)
    return {
        "x-timestamp": ts,
        "x-nonce": nonce,
        "x-signature": sig,
        "x-signature-method": "HMAC-SHA256",
        "x-signature-version": "v1",
        "x-request-id": request_id,
    }


def _poll_get(url: str, timeout_sec: int) -> Dict[str, Any]:
    start = time.time()
    while True:
        try:
            return _http_json("GET", url)
        except Exception:
            if time.time() - start > timeout_sec:
                raise
            time.sleep(0.5)


def main() -> int:
    api_secret = _load_api_secret()

    audio_path = Path(os.environ.get("E2E_AUDIO_PATH", str(DEFAULT_AUDIO_PATH))).resolve()
    video_path = Path(os.environ.get("E2E_VIDEO_PATH", str(DEFAULT_VIDEO_PATH))).resolve()

    # --- ASR ---
    asr_request_id = f"asr-{int(time.time())}"
    asr_headers = _generate_headers(api_secret, request_id="e2e-asr")
    audio_b64 = _read_file_b64(audio_path)
    asr_submit = {
        "request_id": asr_request_id,
        "session_id": "sess-001",
        "segment_index": 0,
        "segment_ts_ms": 0,
        "audio_b64": audio_b64,
        "audio_format": "wav",
        "callback_url": f"{CALLBACK_BASE}/callback/asr",
        "is_last": True,
    }
    asr_ack = _http_json("POST", f"{MAIN_BASE}/api/v1/credit-av-audit/asr/submit", asr_submit, headers=asr_headers)

    try:
        asr_cb = _poll_get(f"{CALLBACK_BASE}/callback/asr/{asr_request_id}", timeout_sec=WAIT_SEC)
    except Exception as e:
        print(f"[WARN] 等待 ASR 回调超时/失败：{e}，尝试 /pull")
        asr_cb = _http_json("POST", f"{MAIN_BASE}/api/v1/credit-av-audit/asr/pull", {"request_id": asr_request_id}, headers=asr_headers)

    # --- Video ---
    video_request_id = f"video-{int(time.time())}"
    video_headers = _generate_headers(api_secret, request_id="e2e-video")
    video_b64 = _read_file_b64(video_path)
    video_submit = {
        "request_id": video_request_id,
        "session_id": "sess-002",
        "segment_index": 0,
        "segment_ts_ms": 0,
        "video_b64": video_b64,
        "video_format": "mp4",
        "task_list": [{"task_id": "t1", "name": "企业大门门头", "desc": "拍清楚门头"}],
        "callback_url": f"{CALLBACK_BASE}/callback/video-guidance",
        "risk_ruleset": "default",
        "is_last": True,
        "ext": {"from": "e2e-client"},
    }
    video_ack = _http_json("POST", f"{MAIN_BASE}/api/v1/credit-av-audit/video-guidance/submit", video_submit, headers=video_headers)

    try:
        video_cb = _poll_get(f"{CALLBACK_BASE}/callback/video-guidance/{video_request_id}", timeout_sec=WAIT_SEC)
    except Exception as e:
        print(f"[WARN] 等待 Video 回调超时/失败：{e}，尝试 /pull")
        video_cb = _http_json("POST", f"{MAIN_BASE}/api/v1/credit-av-audit/video-guidance/pull", {"request_id": video_request_id}, headers=video_headers)

    print("\n=== ASR ACK ===")
    print(json.dumps(asr_ack, ensure_ascii=False, indent=2))
    print("\n=== ASR RESULT ===")
    print(json.dumps(asr_cb, ensure_ascii=False, indent=2))
    print("\n=== VIDEO ACK ===")
    print(json.dumps(video_ack, ensure_ascii=False, indent=2))
    print("\n=== VIDEO RESULT ===")
    print(json.dumps(video_cb, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

