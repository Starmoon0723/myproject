import asyncio
import base64
import hashlib
import hmac
import json
import time
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
import logging
from typing import Any, Dict, Optional

# --- 配置区 ---
API_SECRET = "your-api-secret-here"  # 必须与服务端的 api_secret 保持一致
# SUBMIT_URL = "https://sit-saas-zzbank-ala-vl.qifudigitech.com/api/v1/credit-av-audit/video-guidance/submit"
SUBMIT_URL = "http://127.0.0.1:8000/api/v1/credit-av-audit/video-guidance/submit"
CALLBACK_URL = "https://sit-saas-zzbank-ala-hub.qifudigitech.com/credit-av-audit/callback/video-guidance"
# CALLBACK_URL = "http://127.0.0.1:8001/callback/video-guidance"
VIDEO_PATH = Path("data/IMG_2663_000.mp4")  # 请替换为你本地真实存在的视频路径

# --- 日志配置 ---
# 自动获取当前脚本所在目录 (即 D:\code_project\hengfeng-ssd\video_due_diligence_for_ever_growing\scripts)
SCRIPT_DIR = Path(__file__).parent
LOG_FILE_PATH = SCRIPT_DIR / "script_errors.log"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.ERROR,  # 只记录 ERROR 及以上级别的日志
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

def _generate_headers(api_secret: str, request_id: str) -> dict:
    ts = str(int(time.time()))
    # ts = "1772527958"
    nonce = "BiN-d3WO4_eeoOhTxEpuAldJTar9CsVp"
    string_to_sign = f"{api_secret}{ts}{nonce}"
    digest = hmac.new(api_secret.encode(), string_to_sign.encode(), hashlib.sha256).digest()
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

def _read_video_b64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到测试视频文件：{path}")
    return base64.b64encode(path.read_bytes()).decode()

async def async_http_post(url: str, payload: dict, headers: dict) -> dict:
    """使用 asyncio.to_thread 包装原生的 urllib 实现异步非阻塞请求"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    
    def _make_request():
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"HTTP {e.code} {url}: {body}") from e
            
    return await asyncio.to_thread(_make_request)

def _poll_get(url: str, timeout_sec: int = 60) -> Dict[str, Any]:
    start = time.time()
    while True:
        try:
            return _http_json("GET", url)
        except Exception:
            if time.time() - start > timeout_sec:
                raise
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

async def main():
    for _ in range(10):
        print("1. 正在读取视频并转换为 Base64...")
        video_b64 = _read_video_b64(VIDEO_PATH)
        
        request_id = "aa0b8accc00b444d845216d748c51338"
        headers = _generate_headers(API_SECRET, request_id)
        
        payload = {
            "callback_url": CALLBACK_URL,
            "segment_index": 1,
            "is_last": False,
            "video_b64": video_b64,
            "risk_ruleset": "default-v1",
            "segment_ts_ms": 0,
            "session_id": "176b52ac10ba48c3b1c41605d16bc610",
            "video_format": "mp4",
            "task_list": [
                {"name": "企业大门门头", "task_id": "5411c176051448a2af40a347c5ed33c3", "desc": "拍摄完整企业门头，确保企业名称清晰可识别"},
                {"name": "生产车间", "task_id": "3b68b093ba7f499fb44fcdf139309203", "desc": "车间内有专用生产设备、物料，空间上包含生产标语横幅、安全规范告示墙等"},
                {"name": "GPS地址", "task_id": "0ccf481ec4fd4f9eb72f1f922ccf7e91", "desc": "能识别到拍摄者在车间大门或进入车间后获取到GPS"},
                {"name": "品类", "task_id": "d8dc30c249e645fc9cb00db0d6e59fc8", "desc": "原材料实物，至少1个品类清晰可辨（可结合包装标签）"},
                {"name": "存量规模", "task_id": "25526f4c5a254911a37e97c63cbcb334", "desc": "可计数的原材料堆垛，确保能估算存量（可结合包装数量标签）"},
                {"name": "生产机器设备", "task_id": "49441080e13045d08897b0238929fd08", "desc": "生产线机器设备整体外观，确保设备类型可识别"},
                {"name": "运行状态", "task_id": "a8d80a9e18524cbdaa733fa483d1bea1", "desc": "设备运转情况或操作/监控屏，明确设备当前状态"},
                {"name": "产能", "task_id": "cd659f52da404364883a09da3f3e8621", "desc": "拍摄到机器信息监控屏幕、生产信息监控大屏，开工情况、生产计划、当前产能可识别"},
                {"name": "生产质量", "task_id": "2ec2d8fc7a43465cb4190d9508585759", "desc": "监控大屏或纸质检验报告，需体现质检情况、良品率、残次数量"},
                {"name": "现场作业人员", "task_id": "17ad264edeb54661881853a66ec5c3c5", "desc": "3名以上着工装、作业中（包含生产、监工、调试设备等）的工人"}
            ],
            "request_id": request_id,
            "ext": None
        }

        # 打印请求体结构（隐藏过长的 base64 字符串以便于观察）
        display_payload = payload.copy()
        display_payload["video_b64"] = f"{video_b64[:20]}...[已截断，总长度 {len(video_b64)} 字符]..."
        print("\n=== 发送的完整请求内容 ===")
        print(f"URL: {SUBMIT_URL}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        print(f"Payload: {json.dumps(display_payload, ensure_ascii=False, indent=2)}")

        print("\n2. 正在异步发送 Submit 请求...")
        start_time = time.time()
        try:
            ack_response = await async_http_post(SUBMIT_URL, payload, headers)
            # print("\n=== 收到主服务的 ACK 响应 (说明同步校验通过) ===")
            print(json.dumps(ack_response, ensure_ascii=False, indent=2))
            print(f"请求耗时: {time.time() - start_time:.2f} 秒")
            
            # print("\n提示：此时服务端大模型已在后台运行。")
            # print(f"请检查回调服务器 ({CALLBACK_URL}) 的日志，等待接收服务端主动推送的结果。")
        except Exception as e:
            print(f"\n[报错] 请求失败: {str(1)}")
            logging.error(f"请求提交失败 (URL: {SUBMIT_URL})", exc_info=True)
            # pass
        
        # print(f"走 pull 查看是否进了缓存")
        # f"http://127.0.0.1:8000/api/v1/credit-av-audit/video-guidance/pull",
        pull_req = {"request_id": request_id}
        start_time = time.time()
        while True:
            video_cb = _http_json(
                "POST",
                # f"https://sit-saas-zzbank-ala-vl.qifudigitech.com/api/v1/credit-av-audit/video-guidance/pull",
                f"http://127.0.0.1:8000/api/v1/credit-av-audit/video-guidance/pull",
                pull_req,
                headers=headers,
            )
            if video_cb.get('code') is not None:
                print(json.dumps(video_cb, ensure_ascii=False, indent=2))
                break
            time.sleep(0.2)
        end_time = time.time()
        print(f"pull time: {end_time - start_time:.2f} seconds")
        # video_cb = _http_json(
        #     "POST",
        #     f"https://sit-saas-zzbank-ala-vl.qifudigitech.com/api/v1/credit-av-audit/video-guidance/pull",
        #     pull_req,
        #     headers=headers,
        # )
        # # 打印状态码
        # print(json.dumps(video_cb, ensure_ascii=False, indent=2))
    

    # try:
    #     video_cb = _poll_get(f"http://127.0.0.1:8001/callback/video-guidance/{request_id}", timeout_sec=60)
    # except Exception as e:
    #     print(f"[WARN] 等待 Video 回调超时/失败：{e}，尝试走 pull 查看是否进了缓存")
    #     pull_req = {"request_id": request_id}
    #     video_cb = _http_json(
    #         "POST",
    #         f"http://{MAIN_HOST}:{MAIN_PORT}/api/v1/credit-av-audit/video-guidance/pull",
    #         pull_req,
    #         headers=headers,
    #     )
    # print(json.dumps(video_cb, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 兼容 Windows 系统的 asyncio 策略
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())