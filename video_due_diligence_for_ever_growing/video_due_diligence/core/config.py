import yaml
import os
from typing import Any, Dict


class Config:
    """全局配置类"""
    
    _instance = None
    _config = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            # 如果配置文件不存在，使用默认配置
            self._config = {
                "api": {
                    "secret": "your-api-secret-here",
                    "request_timeout": 300
                },
                "cache": {
                    "default_ttl": 3600,
                    "cleanup_interval": 60
                },
                "server": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "reload": True
                }
            }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置项的值
        :param key_path: 配置项路径，如 "api.secret"
        :param default: 默认值
        :return: 配置项的值
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


# 全局配置实例
config = Config()