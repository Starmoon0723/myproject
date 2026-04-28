# QfinVidCore/utils/env_check.py
"""
环境检查工具，用于检查ffmpeg、torch等依赖项是否正确安装。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from QfinVidCore.utils.runtime import which


def check_ffmpeg() -> bool:
    """
    检查ffmpeg是否已安装且在PATH中。
    
    Returns:
        bool: 如果ffmpeg已安装返回True，否则返回False
    """
    return which("ffmpeg") is not None and which("ffprobe") is not None


def check_torch() -> bool:
    """
    检查torch是否已安装。
    
    Returns:
        bool: 如果torch已安装返回True，否则返回False
    """
    try:
        import torch
        return True
    except ImportError:
        return False


def get_environment_status() -> Dict[str, bool]:
    """
    获取当前环境状态，检查所有依赖项。
    
    Returns:
        Dict[str, bool]: 依赖项状态字典
    """
    return {
        "ffmpeg": check_ffmpeg(),
        "torch": check_torch(),
    }


def validate_environment(minimal: bool = True) -> List[str]:
    """
    验证环境是否满足SDK要求。
    
    Args:
        minimal: 是否仅验证最小依赖项（ffmpeg）
    
    Returns:
        List[str]: 错误信息列表，如果环境正常则返回空列表
    """
    errors = []
    
    if not check_ffmpeg():
        errors.append("ffmpeg未安装或不在PATH中")
    
    if not minimal and not check_torch():
        errors.append("torch未安装")
    
    return errors
