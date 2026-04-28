# QfinVidCore/utils/runtime/hwaccel.py
"""Runtime hwaccel detection and ffmpeg spec injection."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import Any, Dict, Optional

from QfinVidCore.utils.runtime.ffmpeg_runner import get_ffmpeg_bin, run_cmd

_SUPPORTED_BACKENDS = ("cuda", "videotoolbox")


def get_hwaccel_mode() -> str:
    """Return runtime hwaccel mode: auto/off/force."""
    mode = os.environ.get("QFIN_HWACCEL_MODE", "auto").strip().lower()
    if mode in {"auto", "off", "force"}:
        return mode
    return "auto"


def get_hwaccel_backend_override() -> Optional[str]:
    """Read optional backend override from env."""
    backend = os.environ.get("QFIN_HWACCEL_BACKEND", "").strip().lower()
    if backend in _SUPPORTED_BACKENDS:
        return backend
    return None


def _preferred_order() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("videotoolbox", "cuda")
    return ("cuda", "videotoolbox")


@lru_cache(maxsize=1)
def detect_hwaccel_backend() -> Optional[str]:
    """Detect supported hwaccel backend from ffmpeg capabilities."""
    argv = [get_ffmpeg_bin(), "-hide_banner", "-hwaccels"]
    result = run_cmd(argv)
    if not result.ok:
        return None

    text = f"{result.stdout}\n{result.stderr}".lower()
    available: set[str] = set()
    for backend in _SUPPORTED_BACKENDS:
        if re.search(rf"(^|\s){re.escape(backend)}(\s|$)", text, flags=re.MULTILINE):
            available.add(backend)

    if not available:
        return None

    override = get_hwaccel_backend_override()
    if override and override in available:
        return override

    for backend in _preferred_order():
        if backend in available:
            return backend
    return None


def reset_hwaccel_cache() -> None:
    """Clear cached hwaccel probe results (useful for tests)."""
    detect_hwaccel_backend.cache_clear()


def inject_hwaccel_spec(spec: Dict[str, Any], backend: Optional[str]) -> Optional[Dict[str, Any]]:
    """Inject `-hwaccel <backend>` into spec._global when appropriate."""
    if not backend:
        return None

    global_opts = dict(spec.get("_global") or {})
    if "hwaccel" in global_opts or "-hwaccel" in global_opts:
        return None

    injected = dict(spec)
    global_opts["hwaccel"] = backend
    injected["_global"] = global_opts
    return injected

