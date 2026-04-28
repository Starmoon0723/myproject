# QfinVidCore/cognition/__init__.py
"""
认知能力模块。
"""

from __future__ import annotations

# __all__ 只约束 import * 的行为
__all__ = [
    "summarization",
    "consistency_verify",
]

from QfinVidCore.cognition import summarization
from QfinVidCore.cognition import consistency_verify
