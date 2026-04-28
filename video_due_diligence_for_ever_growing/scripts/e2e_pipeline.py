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
from typing import Dict, Any, Optional


ROOT_DIR = str(Path(__file__).resolve().parents[1])
DATA_DIR = Path(ROOT_DIR) / "data"
DEFAULT_AUDIO_PATH = DATA_DIR / "test.wav"
DEFAULT_VIDEO_PATH = DATA_DIR / "IMG_2663_000.mp4"

MAIN_HOST = "127.0.0.1"
MAIN_PORT = 8000
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8001


def _read_proc_output(proc: subprocess.Popen) -> str:
    try:
        if proc.stdout is None:
            return ""
        return proc.stdout.read() or ""
    except Exception:
        return ""


def _wait_port_open(host: str, port: int, timeout_sec: int = 15, proc: Optional[subprocess.Popen] = None) -> None:
    start = time.time()
    while True:
        if proc is not None and proc.poll() is not None:
            out = _read_proc_output(proc)
            raise RuntimeError(f"进程已退出，无法监听端口 {host}:{port}\n{out}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"等待端口开放超时: {host}:{port}")
            time.sleep(0.2)


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
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} {url}: {body}") from e


def _load_api_secret() -> str:
    # 优先从 config.yaml 读（没有也不影响，回退到默认值）
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    try:
        import yaml  # type: ignore  # PyYAML 在 requirements.txt 里
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

    # print(f"digest: {base64.urlsafe_b64encode(digest).decode()}")


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


def _poll_get(url: str, timeout_sec: int = 60) -> Dict[str, Any]:
    start = time.time()
    while True:
        try:
            return _http_json("GET", url)
        except Exception:
            if time.time() - start > timeout_sec:
                raise
            time.sleep(0.2)


def main() -> int:
    api_secret = _load_api_secret()

    env = os.environ.copy()
    cwd = ROOT_DIR
    py = sys.executable

    audio_path = Path(os.environ.get("E2E_AUDIO_PATH", str(DEFAULT_AUDIO_PATH))).resolve()
    video_path = Path(os.environ.get("E2E_VIDEO_PATH", str(DEFAULT_VIDEO_PATH))).resolve()

    callback_cmd = [py, "-m", "uvicorn", "callback_app:app", "--host", CALLBACK_HOST, "--port", str(CALLBACK_PORT), "--log-level", "warning"]
    main_cmd = [py, "-m", "uvicorn", "video_due_diligence.main:app", "--host", MAIN_HOST, "--port", str(MAIN_PORT), "--log-level", "warning"]

    callback_proc = None
    main_proc = None
    try:
        print(f"[1/4] 启动 mock 回调服务: {CALLBACK_HOST}:{CALLBACK_PORT}")
        callback_proc = subprocess.Popen(callback_cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _wait_port_open(CALLBACK_HOST, CALLBACK_PORT, timeout_sec=20, proc=callback_proc)

        print(f"[2/4] 启动主服务: {MAIN_HOST}:{MAIN_PORT}")
        main_proc = subprocess.Popen(main_cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _wait_port_open(MAIN_HOST, MAIN_PORT, timeout_sec=20, proc=main_proc)

        # --- ASR submit ---
        print("[3/4] 发起 ASR submit 并等待回调落库")
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
            "callback_url": f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback/asr",
            "is_last": True,
        }
        asr_ack = _http_json("POST", f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/asr/submit", asr_submit, headers=asr_headers)

        try:
            asr_cb = _poll_get(f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback/asr/{asr_request_id}", timeout_sec=60)
        except Exception as e:
            # 如果模型推理慢/回调失败，可能会走 pull 缓存；这里做兜底展示，方便定位
            print(f"[WARN] 等待 ASR 回调超时/失败：{e}，尝试走 pull 查看是否进了缓存")
            pull_req = {"request_id": asr_request_id}
            asr_cb = _http_json(
                "POST",
                f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/asr/pull",
                pull_req,
                headers=asr_headers,
            )

        print("\n=== ASR ACK ===")
        print(json.dumps(asr_ack, ensure_ascii=False, indent=2))
        print("\n=== ASR CALLBACK ===")
        print(json.dumps(asr_cb, ensure_ascii=False, indent=2))

        # --- Video submit ---
        print("[4/4] 发起 Video submit 并等待回调落库")
        video_request_id = f"video-{int(time.time())}"
        video_headers = _generate_headers(api_secret, request_id="e2e-video")
        # video_b64 = _read_file_b64(video_path)
        video_b64 = str(video_path)
        default_task_names = ['企业大门门头', '生产车间', '品类', '存量规模', '生产机器设备', '运行状态','产能', '生产质量', '现场作业人员', '出勤班表']
        task_list = [{"task_id": f"T{i+1}", "name": n, "desc": f"关注 {n}"} for i, n in enumerate(default_task_names)]
        video_submit = {
            "request_id": video_request_id,
            "session_id": "sess-002",
            "segment_index": 0,
            "segment_ts_ms": 0,
            "video_b64": video_b64,
            "video_format": "mp4",
            "task_list": task_list,
            "callback_url": f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback/video-guidance",
            "risk_ruleset": "default-v1",
            "is_last": False,
            "ext": {
                "loan_id": "loan-123",
                "customer_id": "customer-456",
            },
        }
        video_ack = _http_json("POST", f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/video-guidance/submit", video_submit, headers=video_headers)

        try:
            video_cb = _poll_get(f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback/video-guidance/{video_request_id}", timeout_sec=60)
        except Exception as e:
            print(f"[WARN] 等待 Video 回调超时/失败：{e}，尝试走 pull 查看是否进了缓存")
            pull_req = {"request_id": video_request_id}
            video_cb = _http_json(
                "POST",
                f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/video-guidance/pull",
                pull_req,
                headers=video_headers,
            )

        print("\n=== VIDEO ACK ===")
        print(json.dumps(video_ack, ensure_ascii=False, indent=2))
        print("\n=== VIDEO CALLBACK ===")
        print(json.dumps(video_cb, ensure_ascii=False, indent=2))

        return 0
    finally:
        # 关键修复：
        # - 之前在 terminate 之前调用 stdout.read() 会阻塞（子进程没退出 read 会一直等 EOF），导致脚本“打印完回调但不结束”
        # - 改为：先 terminate -> communicate(timeout) 收集输出；超时则 kill 再收集
        def _shutdown_proc(proc: Optional[subprocess.Popen], title: str) -> None:
            if proc is None:
                return
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                out, _ = proc.communicate(timeout=3)
            except Exception:
                try:
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                try:
                    out, _ = proc.communicate(timeout=3)
                except Exception:
                    out = ""
            if out and str(out).strip():
                print(f"\n=== {title} OUTPUT ===")
                print(out)

        _shutdown_proc(main_proc, "MAIN SERVER")
        _shutdown_proc(callback_proc, "CALLBACK SERVER")


if __name__ == "__main__":
    raise SystemExit(main())

