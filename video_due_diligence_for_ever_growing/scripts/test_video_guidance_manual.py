import base64
import hashlib
import hmac
import json
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

# === 配置区域 ===
ROOT_DIR = Path(__file__).resolve().parents[1]
# 视频文件路径 (请修改为你本地的真实视频路径)
VIDEO_PATH = str(ROOT_DIR / "data/IMG_2663_000.mp4")

# 尝试从 config.yaml 加载 API_SECRET
try:
    import yaml
    config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        API_SECRET = config.get("api", {}).get("secret", "your-api-secret-here")
except Exception as e:
    print(f"Warning: 无法加载 config.yaml ({e})，使用默认密钥")
    API_SECRET = "your-api-secret-here"

# 接口地址
BASE_URL = "https://sit-saas-zzbank-ala-vl.qifudigitech.com/api/v1/credit-av-audit/video-guidance"
SUBMIT_URL = f"{BASE_URL}/submit"
PULL_URL = f"{BASE_URL}/pull"

# === 辅助函数 ===

def _read_file_b64(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"找不到视频文件: {path_str}")
    print(f"正在读取视频文件: {path_str} (大小: {path.stat().st_size / 1024 / 1024:.2f} MB)...")
    return base64.b64encode(path.read_bytes()).decode('utf-8')

def _generate_signature(api_secret: str, timestamp: str, nonce: str) -> str:
    string_to_sign = f"{api_secret}{timestamp}{nonce}"
    digest = hmac.new(api_secret.encode(), string_to_sign.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode()

def _generate_headers(api_secret: str, request_id: str) -> dict:
    ts = str(int(time.time()))
    nonce = uuid.uuid4().hex[:8]
    sig = _generate_signature(api_secret, ts, nonce)
    return {
        "Content-Type": "application/json",
        "x-timestamp": ts,
        "x-nonce": nonce,
        "x-signature": sig,
        "x-signature-method": "HMAC-SHA256",
        "x-signature-version": "v1",
        "x-request-id": request_id,
    }

def _http_post(url: str, payload: dict, headers: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"请求失败 [{e.code}]: {url}")
        print(f"响应内容: {body}")
        raise

# === 主逻辑 ===

def main():
    # 1. 准备请求数据
    print("=== 1. 准备请求数据 ===")
    
    # 生成新的 ID 避免重复请求错误
    request_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    print(f"生成的 Request ID: {request_id}")
    print(f"生成的 Session ID: {session_id}")

    try:
        video_b64 = _read_file_b64(VIDEO_PATH)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请修改脚本中的 VIDEO_PATH 变量为有效的视频文件路径。")
        return

    # 构造 Payload (基于用户提供的 JSON)
    payload = {
        "callback_url": "https://sit-saas-zzbank-ala-hub.qifudigitech.com/credit-av-audit/callback/video-guidance",
        "segment_index": 1,  # 用户提供的示例是 1，这里保持一致。通常单个文件上传应为 0。
        "is_last": True,     # 强制设为 True 以触发服务端处理（用户示例为 false，可能导致不触发回调）
        "video_b64": video_b64,
        "risk_ruleset": "default-v1",
        "segment_ts_ms": 0,
        "session_id": session_id,
        "video_format": "mp4",
        "task_list": [
            {
                "name": "企业大门门头",
                "task_id": "5411c176051448a2af40a347c5ed33c3",
                "desc": "拍摄完整企业门头，确保企业名称清晰可识别"
            },
            {
                "name": "生产车间",
                "task_id": "3b68b093ba7f499fb44fcdf139309203",
                "desc": "车间内有专用生产设备、物料，空间上包含生产标语横幅、安全规范告示墙等"
            },
            {
                "name": "GPS地址",
                "task_id": "0ccf481ec4fd4f9eb72f1f922ccf7e91",
                "desc": "能识别到拍摄者在车间大门或进入车间后获取到GPS"
            },
            {
                "name": "品类",
                "task_id": "d8dc30c249e645fc9cb00db0d6e59fc8",
                "desc": "原材料实物，至少1个品类清晰可辨（可结合包装标签）"
            },
            {
                "name": "存量规模",
                "task_id": "25526f4c5a254911a37e97c63cbcb334",
                "desc": "可计数的原材料堆垛，确保能估算存量（可结合包装数量标签）"
            },
            {
                "name": "生产机器设备",
                "task_id": "49441080e13045d08897b0238929fd08",
                "desc": "生产线机器设备整体外观，确保设备类型可识别"
            },
            {
                "name": "运行状态",
                "task_id": "a8d80a9e18524cbdaa733fa483d1bea1",
                "desc": "设备运转情况或操作/监控屏，明确设备当前状态"
            },
            {
                "name": "产能",
                "task_id": "cd659f52da404364883a09da3f3e8621",
                "desc": "拍摄到机器信息监控屏幕、生产信息监控大屏，开工情况、生产计划、当前产能可识别"
            },
            {
                "name": "生产质量",
                "task_id": "2ec2d8fc7a43465cb4190d9508585759",
                "desc": "监控大屏或纸质检验报告，需体现质检情况、良品率、残次数量"
            },
            {
                "name": "现场作业人员",
                "task_id": "17ad264edeb54661881853a66ec5c3c5",
                "desc": "3名以上着工装、作业中（包含生产、监工、调试设备等）的工人"
            }
        ],
        "request_id": request_id
    }

    # 2. 发送 Submit 请求
    print("\n=== 2. 发送 Submit 请求 ===")
    headers = _generate_headers(API_SECRET, request_id)
    try:
        start_time = time.time()
        submit_resp = _http_post(SUBMIT_URL, payload, headers)
        print(f"请求耗时: {time.time() - start_time:.2f}s")
        print("Submit 响应:")
        print(json.dumps(submit_resp, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Submit 请求异常: {e}")
        return

    # 3. 轮询 Pull 接口获取异步结果
    print("\n=== 3. 轮询获取异步结果 (Pull) ===")
    print(f"将尝试调用 Pull 接口: {PULL_URL}")
    print("注意：如果 Pull 接口不可用或返回 404，请检查服务端是否支持 Pull 模式，或直接查看 Callback URL 的接收情况。")
    
    max_retries = 10
    retry_interval = 2 # 秒

    pull_payload = {"request_id": request_id}
    # Pull 接口的 Request ID 通常可以是查询用的 ID，或者 generate 一个新的 header ID
    # 这里我们复用 header 生成逻辑，但 request_id 依然是我们要查的那个
    
    for i in range(max_retries):
        print(f"尝试第 {i+1}/{max_retries} 次 Pull...")
        pull_headers = _generate_headers(API_SECRET, f"pull-{uuid.uuid4().hex[:8]}")
        
        try:
            pull_resp = _http_post(PULL_URL, pull_payload, pull_headers)
            
            # 假设服务端返回结构中有 code/status 等字段指示状态
            # 如果直接返回结果，打印并退出
            # 这里简单判断：如果返回了非空数据，且不像是一个"处理中"的错误，就认为是结果
            
            print("Pull 响应:")
            print(json.dumps(pull_resp, ensure_ascii=False, indent=2))
            
            # 简单的判断逻辑：如果响应里包含了结果数据（比如 answer 字段），则认为成功
            if "data" in pull_resp and pull_resp["data"]:
                 # 根据实际 API 结构，这里可能需要调整判断逻辑
                 # 假设 data 非空即为有结果
                 print("\n>>> 获取到异步结果！")
                 break
            
            # 如果 API 明确返回 "processing" 或类似状态，继续轮询
            # 这里假设如果没结果，可能 data 是空的或者 null
            
        except urllib.error.HTTPError as e:
            # 如果是 404，可能是还没生成结果，或者接口不对
            if e.code == 404:
                print("Pull 返回 404 (可能结果未生成或接口不存在)")
            else:
                print(f"Pull 失败: {e}")
        except Exception as e:
            print(f"Pull 异常: {e}")
        
        time.sleep(retry_interval)
    else:
        print("\n[超时] 未能通过 Pull 接口获取到最终结果。请检查 Callback URL 接收到的数据。")

if __name__ == "__main__":
    main()
