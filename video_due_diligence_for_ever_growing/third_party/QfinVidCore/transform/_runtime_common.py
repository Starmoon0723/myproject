# QfinVidCore/transform/_runtime_common.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.utils.runtime.ffmpeg_runner import run_ffmpeg_spec, run_ffprobe_spec
from QfinVidCore.utils.runtime.hwaccel import (
    detect_hwaccel_backend,
    get_hwaccel_mode,
    inject_hwaccel_spec,
)

Pathlike = Union[str, Path]


def as_path(value: Pathlike) -> Path:
    return value if isinstance(value, Path) else Path(value)


def resolve_input_path(input_video: Union[VideoEntity, Pathlike]) -> Path:
    if isinstance(input_video, VideoEntity):
        return Path(input_video.uri).resolve()
    return as_path(input_video).resolve()


def run_ffmpeg_or_raise(operation: str, spec: Dict[str, Any], *, timeout: float | None = None) -> None:
    def _raise_failure(prefix: str, result: Any) -> None:
        cmd = " ".join(result.cmd)
        details = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
        raise RuntimeError(f"{operation} failed ({prefix}): {details}\ncmd: {cmd}")

    mode = get_hwaccel_mode()
    if mode == "off":
        result = run_ffmpeg_spec(spec, timeout=timeout)
        if result.ok:
            return
        _raise_failure("cpu", result)

    backend = detect_hwaccel_backend()
    hw_spec = inject_hwaccel_spec(spec, backend)

    if mode == "force":
        if hw_spec is None:
            if backend is None:
                raise RuntimeError(
                    f"{operation} failed (hwaccel): no supported backend detected while mode=force"
                )
            # User already specified hwaccel in spec._global.
            result = run_ffmpeg_spec(spec, timeout=timeout)
            if result.ok:
                return
            _raise_failure("hwaccel(force)", result)
        result = run_ffmpeg_spec(hw_spec, timeout=timeout)
        if result.ok:
            return
        _raise_failure("hwaccel(force)", result)

    # auto mode
    if hw_spec is not None:
        hw_result = run_ffmpeg_spec(hw_spec, timeout=timeout)
        if hw_result.ok:
            return

        cpu_result = run_ffmpeg_spec(spec, timeout=timeout)
        if cpu_result.ok:
            return

        hw_cmd = " ".join(hw_result.cmd)
        hw_details = hw_result.stderr.strip() or hw_result.stdout.strip() or f"returncode={hw_result.returncode}"
        cpu_cmd = " ".join(cpu_result.cmd)
        cpu_details = (
            cpu_result.stderr.strip() or cpu_result.stdout.strip() or f"returncode={cpu_result.returncode}"
        )
        raise RuntimeError(
            f"{operation} failed (hwaccel then cpu fallback):\n"
            f"[hwaccel] {hw_details}\ncmd: {hw_cmd}\n"
            f"[cpu] {cpu_details}\ncmd: {cpu_cmd}"
        )

    result = run_ffmpeg_spec(spec, timeout=timeout)
    if result.ok:
        return
    _raise_failure("cpu", result)


def run_ffprobe_or_raise(operation: str, spec: Dict[str, Any], *, timeout: float | None = None) -> str:
    result = run_ffprobe_spec(spec, timeout=timeout)
    if result.ok:
        return result.stdout
    cmd = " ".join(result.cmd)
    details = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
    raise RuntimeError(f"{operation} failed: {details}\ncmd: {cmd}")


def run_ffprobe_json_or_raise(
    operation: str,
    spec: Dict[str, Any],
    *,
    timeout: float | None = None,
) -> Dict[str, Any]:
    out = run_ffprobe_or_raise(operation, spec, timeout=timeout).strip()
    return json.loads(out or "{}")
