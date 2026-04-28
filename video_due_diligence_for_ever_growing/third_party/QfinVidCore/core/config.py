# QfinVidCore/core/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

Pathlike = Union[str, Path]


class ConfigManager:
    """
    配置管理器，用于加载和访问配置文件
    单例模式，确保全局只有一个配置实例
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        初始化配置管理器，加载默认配置文件
        """
        # 加载默认配置文件
        config_path = Path(__file__).parent.parent / "config.yaml"
        self._config = load_config(config_path)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置项，支持点号分隔的路径
        
        Args:
            key_path: 配置项路径，如 "core.log_level"
            default: 默认值，如果配置项不存在则返回
            
        Returns:
            配置项的值
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update(self, config_dict: Dict[str, Any]):
        """
        更新配置
        
        Args:
            config_dict: 要更新的配置字典
        """
        self._config.update(config_dict)
    
    def reload(self):
        """
        重新加载配置文件
        """
        self._initialize()


def load_config(path: Optional[Pathlike] = None) -> Dict[str, Any]:
    """
    加载 yaml 配置（如果未安装 PyYAML，则返回空 dict）。
    文档中提到 load_config() # yaml。:contentReference[oaicite:2]{index=2}
    """
    if path is None:
        return {}

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在：{p}")

    try:
        import yaml  # type: ignore
    except Exception:
        # 允许不装 yaml，也能跑核心能力
        return {}

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是 dict")
    return data


# 创建全局配置管理器实例
config_manager = ConfigManager()


# 导出常用函数
def get_config(key_path: str, default: Any = None) -> Any:
    """
    获取配置项的便捷函数
    
    Args:
        key_path: 配置项路径
        default: 默认值
        
    Returns:
        配置项的值
    """
    return config_manager.get(key_path, default)
