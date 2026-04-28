import threading
import time
from typing import Dict, Optional, Any
from ..core.config import config


class ResultCache(dict):
    """结果缓存类，继承自dict，用于存储ASR和视频尽调的处理结果"""
    
    def __init__(self):
        """
        初始化结果缓存
        """
        super().__init__()
        self._lock = threading.Lock()
        self._cleanup_interval = config.get("cache.cleanup_interval", 60)
        self._default_ttl = config.get("cache.default_ttl", 3600)
        
        # 启动后台清理线程
        self._start_cleanup_thread()
    
    def store_result(self, request_id: str, result: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        存储处理结果
        :param request_id: 请求ID
        :param result: 处理结果
        :param ttl: 过期时间（秒），如果为None则使用默认值
        """
        if ttl is None:
            ttl = self._default_ttl
            
        now = time.time()
        expire_time = now + float(ttl or 0)
        
        with self._lock:
            self[request_id] = {
                "result": result,
                "cached_at": now,
                "expire_at": expire_time
            }
    
    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        获取处理结果
        :param request_id: 请求ID
        :return: 处理结果，如果不存在则返回None
        """
        with self._lock:
            if request_id not in self:
                return None

            item = self[request_id]
            return {
                "data": item["result"],
                "trace": {
                    "cached_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(item["cached_at"])),
                    "expired_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(item["expire_at"])),
                },
            }
    
    def remove_result(self, request_id: str) -> bool:
        """
        删除处理结果
        :param request_id: 请求ID
        :return: 如果删除成功返回True，否则返回False
        """
        with self._lock:
            if request_id in self:
                del self[request_id]
                return True
            return False
    
    def _start_cleanup_thread(self) -> None:
        """启动后台清理线程"""
        cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self) -> None:
        """后台定时清理过期的缓存项"""
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                # 在锁外遍历字典可能会有问题，但在锁内操作整个字典可能会影响性能
                # 这里我们接受可能的小误差，优先保证性能
                with self._lock:
                    for key, value in self.items():
                        if current_time > value["expire_at"]:
                            expired_keys.append(key)
                
                # 删除过期项
                with self._lock:
                    for key in expired_keys:
                        del self[key]
                
                # 等待下一个清理周期
                time.sleep(self._cleanup_interval)
            except Exception:
                # 防止线程因为异常而退出
                time.sleep(self._cleanup_interval)


# 结果缓存单例：整个进程共用，避免多实例 + 多清理线程
result_cache = ResultCache()