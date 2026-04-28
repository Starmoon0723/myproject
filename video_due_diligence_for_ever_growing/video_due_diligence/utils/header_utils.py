from fastapi import Request, HTTPException
from typing import Dict, Any
from starlette.datastructures import Headers
from ..core.config import config
import base64
import hashlib
import hmac
import time


class HeaderHandler:
    """处理通用Header的工具类"""
    
    def __init__(self, api_secret: str = ""):
        self.api_secret = api_secret or config.get("api.secret", "")
        self.request_timeout = config.get("api.request_timeout", 300)
    
    async def validate_headers(self, request: Request) -> Dict[str, Any]:
        """验证请求头中的签名和其他必要字段"""
        # 获取所有必需的header
        headers = request.headers
        required_headers = [
            "x-timestamp", "x-nonce", "x-signature",
            "x-signature-method", "x-signature-version", "x-request-id"
        ]
        
        # 检查必需的header是否存在
        for header in required_headers:
            if header not in headers:
                raise HTTPException(status_code=400, detail=f"Missing header: {header}")
        
        # 验证签名方法和版本
        if headers["x-signature-method"] != "HMAC-SHA256":
            raise HTTPException(status_code=400, detail="Invalid signature method")
        
        if headers["x-signature-version"] != "v1":
            raise HTTPException(status_code=400, detail="Invalid signature version")
        
        # 验证时间戳（防止重放攻击）
        try:
            timestamp = int(headers["x-timestamp"])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid timestamp")
        current_time = int(time.time())
        if abs(current_time - timestamp) > self.request_timeout:
            raise HTTPException(status_code=401, detail="Request expired")
        
        # 验证签名
        if not self._verify_signature(headers):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        return {
            "timestamp": timestamp,
            "nonce": headers["x-nonce"],
            "request_id": headers["x-request-id"]
        }
    
    def _verify_signature(self, headers: Headers) -> bool:
        """验证签名"""
        timestamp = headers["x-timestamp"]
        nonce = headers["x-nonce"]
        
        # 构造签名字符串
        string_to_sign = f"{self.api_secret}{timestamp}{nonce}"
        
        # 计算签名
        expected_signature = base64.urlsafe_b64encode(
            hmac.new(
                self.api_secret.encode(),
                string_to_sign.encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        
        # 比较签名
        return hmac.compare_digest(expected_signature, headers["x-signature"])


class DataDeserializer:
    """处理数据反序列化的工具类"""
    
    @staticmethod
    def base64_to_bytes(base64_str: str) -> bytes:
        """将base64字符串转换为字节数据"""
        try:
            # 允许标准 base64；严格校验非法字符
            return base64.b64decode(base64_str, validate=True)
        except Exception:
            # 兼容部分调用方可能传 urlsafe base64
            try:
                padded = base64_str + "=" * (-len(base64_str) % 4)
                return base64.urlsafe_b64decode(padded)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
    
    @staticmethod
    def base64_to_audio(base64_str: str) -> bytes:
        """将base64字符串转换为音频数据"""
        return DataDeserializer.base64_to_bytes(base64_str)
    
    @staticmethod
    def base64_to_video(base64_str: str) -> bytes:
        """将base64字符串转换为视频数据"""
        return DataDeserializer.base64_to_bytes(base64_str)