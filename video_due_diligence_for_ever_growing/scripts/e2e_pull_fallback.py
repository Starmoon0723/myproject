"""
E2E 脚本：验证 vibe.md 第三条“回调失败 -> 写缓存 -> pull 拉取”的兜底链路。

核心思路：
1) 只启动主服务（video_due_diligence.main:app），不启动 callback 服务
2) submit 时把 callback_url 指向非本地地址（默认 config.yaml: api.allow_external_callback=false）
   后台任务会拒绝外网回调并将 payload 存入 result_cache
3) 轮询调用 pull 接口拉取结果；成功后再拉一次验证已 purged（应 404）

运行方式（项目根目录）：
  python scripts/e2e_pull_fallback.py
"""

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


ROOT_DIR = str(Path(__file__).resolve().parents[1])
DATA_DIR = Path(ROOT_DIR) / "data"
DEFAULT_AUDIO_PATH = DATA_DIR / "test.wav"
DEFAULT_VIDEO_PATH = DATA_DIR / "IMG_2663_000.mp4"

MAIN_HOST = "127.0.0.1"
MAIN_PORT = 8000


def _wait_port_open(host: str, port: int, timeout_sec: int = 15, proc: Optional[subprocess.Popen] = None) -> None:
    start = time.time()
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"进程已退出，无法监听端口 {host}:{port}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"等待端口开放超时: {host}:{port}")
            time.sleep(0.2)


def _load_api_secret() -> str:
    config_path = os.path.join(ROOT_DIR, "config.yaml")
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


def _http_json_status(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, Any]]:
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
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return int(resp.status), (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
            parsed = {"raw": body}
        return int(e.code), parsed


def _poll_pull_ok(url: str, headers: Dict[str, str], timeout_sec: int = 30) -> Dict[str, Any]:
    start = time.time()
    last_body: Dict[str, Any] = {}
    while True:
        status, body = _http_json_status("POST", url, payload=last_body or None, headers=headers)
        # pull 接口入参固定：{"request_id": "..."}，调用方会在 main() 里拼好 URL 并传入 payload
        # 这里把 payload 复用为 last_body 只是避免每次创建 dict；main() 会在第一次前设置
        if status == 200 and int(body.get("code", -1)) == 0:
            return body
        if status == 404 and int(body.get("code", -1)) == 1404:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"等待 pull 结果超时: {body}")
            time.sleep(0.2)
            continue
        raise RuntimeError(f"pull 异常: HTTP {status} body={body}")


def main() -> int:
    api_secret = _load_api_secret()

    env = os.environ.copy()
    cwd = ROOT_DIR
    py = sys.executable

    audio_path = Path(os.environ.get("E2E_AUDIO_PATH", str(DEFAULT_AUDIO_PATH))).resolve()
    video_path = Path(os.environ.get("E2E_VIDEO_PATH", str(DEFAULT_VIDEO_PATH))).resolve()

    # 故意使用非本地 callback_url，触发“external callback disabled”从而写缓存
    # 不会真实请求外网（后台任务会提前拒绝）
    asr_callback_url = "http://example.com/callback/asr"
    video_callback_url = "http://example.com/callback/video-guidance"

    main_cmd = [py, "-m", "uvicorn", "video_due_diligence.main:app", "--host", MAIN_HOST, "--port", str(MAIN_PORT), "--log-level", "warning"]

    main_proc: Optional[subprocess.Popen] = None
    try:
        print(f"[1/3] 启动主服务: {MAIN_HOST}:{MAIN_PORT}")
        main_proc = subprocess.Popen(main_cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _wait_port_open(MAIN_HOST, MAIN_PORT, timeout_sec=20, proc=main_proc)

        # --- ASR submit（回调失败 -> pull） ---
        print("[2/3] 发起 ASR submit（回调失败后将从 pull 拉取）")
        asr_request_id = f"asr-pull-{int(time.time())}"
        asr_headers = _generate_headers(api_secret, request_id="e2e-asr-pull")
        audio_b64 = _read_file_b64(audio_path)
        asr_submit = {
            "request_id": asr_request_id,
            "session_id": "sess-pull-001",
            "segment_index": 0,
            "segment_ts_ms": 0,
            "audio_b64": audio_b64,
            "audio_format": "wav",
            "callback_url": asr_callback_url,
            "is_last": True,
        }
        s, asr_ack = _http_json_status("POST", f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/asr/submit", asr_submit, headers=asr_headers)
        if s != 200:
            raise RuntimeError(f"ASR submit 失败: HTTP {s} body={asr_ack}")

        asr_pull_url = f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/asr/pull"
        # 第一次请求 payload 作为轮询入参
        pull_payload = {"request_id": asr_request_id}
        start = time.time()
        while True:
            status, body = _http_json_status("POST", asr_pull_url, payload=pull_payload, headers=asr_headers)
            if status == 200 and int(body.get("code", -1)) == 0:
                asr_pull = body
                break
            if status == 404 and int(body.get("code", -1)) == 1404:
                if time.time() - start > 60:
                    raise TimeoutError(f"等待 ASR pull 超时: {body}")
                time.sleep(0.2)
                continue
            raise RuntimeError(f"ASR pull 异常: HTTP {status} body={body}")

        # 再拉一次应 404（已 purged）
        status2, body2 = _http_json_status("POST", asr_pull_url, payload=pull_payload, headers=asr_headers)

        # --- Video submit（回调失败 -> pull） ---
        print("[3/3] 发起 Video submit（回调失败后将从 pull 拉取）")
        video_request_id = f"video-pull-{int(time.time())}"
        video_headers = _generate_headers(api_secret, request_id="e2e-video-pull")
        video_b64 = _read_file_b64(video_path)
        default_task_names = ['企业大门门头', '生产车间', '品类', '存量规模', '生产机器设备', '运行状态','产能', '生产质量', '现场作业人员', '出勤班表']
        task_list = [{"task_id": f"t{i+1}", "name": n, "desc": f"关注 {n}"} for i, n in enumerate(default_task_names)]
        video_submit = {
            "request_id": video_request_id,
            "session_id": "sess-pull-002",
            "segment_index": 0,
            "segment_ts_ms": 0,
            "video_b64": video_b64,
            "video_format": "mp4",
            "task_list": task_list,
            "callback_url": video_callback_url,
            "risk_ruleset": "default",
            "is_last": True,
            "ext": {"from": "e2e_pull_fallback"},
        }
        s, video_ack = _http_json_status(
            "POST",
            f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/video-guidance/submit",
            video_submit,
            headers=video_headers,
        )
        if s != 200:
            raise RuntimeError(f"Video submit 失败: HTTP {s} body={video_ack}")

        video_pull_url = f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/video-guidance/pull"
        pull_payload2 = {"request_id": video_request_id}
        start = time.time()
        while True:
            status, body = _http_json_status("POST", video_pull_url, payload=pull_payload2, headers=video_headers)
            if status == 200 and int(body.get("code", -1)) == 0:
                video_pull = body
                break
            if status == 404 and int(body.get("code", -1)) == 1404:
                if time.time() - start > 90:
                    raise TimeoutError(f"等待 Video pull 超时: {body}")
                time.sleep(0.2)
                continue
            raise RuntimeError(f"Video pull 异常: HTTP {status} body={body}")

        status3, body3 = _http_json_status("POST", video_pull_url, payload=pull_payload2, headers=video_headers)

        print("\n=== ASR ACK ===")
        print(json.dumps(asr_ack, ensure_ascii=False, indent=2))
        print("\n=== ASR PULL (OK) ===")
        print(json.dumps(asr_pull, ensure_ascii=False, indent=2))
        print("\n=== ASR PULL (2nd, expect 404) ===")
        print(json.dumps({"http_status": status2, "body": body2}, ensure_ascii=False, indent=2))

        print("\n=== VIDEO ACK ===")
        print(json.dumps(video_ack, ensure_ascii=False, indent=2))
        print("\n=== VIDEO PULL (OK) ===")
        print(json.dumps(video_pull, ensure_ascii=False, indent=2))
        print("\n=== VIDEO PULL (2nd, expect 404) ===")
        print(json.dumps({"http_status": status3, "body": body3}, ensure_ascii=False, indent=2))

        return 0
    finally:
        if main_proc is None:
            return
        try:
            if main_proc.poll() is None:
                main_proc.terminate()
        except Exception:
            pass
        try:
            out, _ = main_proc.communicate(timeout=3)
        except Exception:
            try:
                if main_proc.poll() is None:
                    main_proc.kill()
            except Exception:
                pass
            try:
                out, _ = main_proc.communicate(timeout=3)
            except Exception:
                out = ""
        if out and str(out).strip():
            print("\n=== MAIN SERVER OUTPUT ===")
            print(out)


if __name__ == "__main__":
    raise SystemExit(main())

