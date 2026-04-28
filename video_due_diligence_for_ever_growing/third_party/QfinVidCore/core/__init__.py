# QfinVidCore/__init__.py
"""
QfinVidCore - 视频分析/处理 SDK

设计目标：
- 以“视频”为输入，对视频进行编码/时间/空间/内容等变换处理
- 保持导入轻量：import QfinVidCore 不应触发 ffmpeg/环境检查等重操作
- 提供清晰、稳定的对外 API（后续可逐步补全 perception/cognition 等能力）
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "__version__",
    "VideoEntity",
    "load_config",
]

# ---- version ----
def _resolve_version() -> str:
    # 1) 优先使用项目内的 core.version
    try:
        vmod = import_module("QfinVidCore.core.version")
        v = getattr(vmod, "sdk_version", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass

    # 2) 若以 pip 包形式安装，尝试读取包元数据版本
    try:
        from importlib.metadata import PackageNotFoundError, version  # py>=3.8

        try:
            return version("QfinVidCore")
        except PackageNotFoundError:
            return "0.0.0"
    except Exception:
        return "0.0.0"


__version__ = _resolve_version()


# ---- public re-exports (lightweight only) ----
try:
    from QfinVidCore.core.VideoEntity import VideoEntity  # noqa: F401
except Exception as e:  # pragma: no cover
    # 不在 import 阶段强制失败，避免开发期“半成品”阻断；真正使用时会报更具体错误
    VideoEntity = None  # type: ignore[assignment]
    _video_entity_import_error = e


def load_config(*args: Any, **kwargs: Any):
    """
    读取 SDK 配置（yaml 等），延迟导入以保持包导入轻量。
    """
    mod = import_module("QfinVidCore.core.config")
    return mod.load_config(*args, **kwargs)


# ---- optional lazy attributes ----
def __getattr__(name: str) -> Any:
    """
    延迟加载较重/可选模块，避免 import QfinVidCore 时引入 ffmpeg 检查等副作用。
    例如：
      from QfinVidCore import transform
      from QfinVidCore import utils
    """
    if name in {"core", "utils", "transform", "perception", "cognition"}:
        return import_module(f"QfinVidCore.{name}")

    if name == "VideoEntity":
        # 如果上面导入失败，这里给出更明确的错误提示
        if VideoEntity is None:  # type: ignore[name-defined]
            raise ImportError(
                "无法导入 VideoEntity（QfinVidCore.core.VideoEntity）。"
                "请检查文件是否存在、依赖是否正确。"
            ) from globals().get("_video_entity_import_error")
        return VideoEntity

    raise AttributeError(f"module 'QfinVidCore' has no attribute '{name}'")
