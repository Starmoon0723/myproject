# QfinVidCore/utils/runtime/ffmpeg_runner.py
"""
Unified ffmpeg/ffprobe runner for QfinVidCore.

Goals:
- Cross-platform (Windows/Linux/macOS)
- Dict(spec) -> argv(list[str]) with input/output grouping (order-sensitive)
- Strong error messages (return stdout/stderr)
- Allow overriding ffmpeg/ffprobe binaries via env vars: FFMPEG_BIN, FFPROBE_BIN
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]


# ---------------------------
# Result type
# ---------------------------

@dataclass
class CmdResult:
    ok: bool
    returncode: int
    cmd: List[str]
    stdout: str
    stderr: str


# ---------------------------
# Platform helpers
# ---------------------------

def is_windows() -> bool:
    return os.name == "nt"


def null_sink() -> str:
    """Platform-specific null sink for ffmpeg/ffprobe outputs."""
    return "NUL" if is_windows() else "/dev/null"

# 1. 在终端运行 export FFMPEG_BIN=/path/to/custom/ffmpeg 即可指定自定义ffmpeg/ffprobe的路径，硬件加速的或者其他的ffmpeg/ffprobe版本
# 2. 比硬编码更健壮，一点也不业余
def get_ffmpeg_bin() -> str:
    """
    Resolve ffmpeg executable.
    Priority:
      1) env FFMPEG_BIN
      2) system PATH via shutil.which
      3) fallback to 'ffmpeg' (may fail later with clear error)
    """
    env = os.environ.get("FFMPEG_BIN")
    if env:
        return env
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def get_ffprobe_bin() -> str:
    env = os.environ.get("FFPROBE_BIN")
    if env:
        return env
    found = shutil.which("ffprobe")
    return found or "ffprobe"


# ---------------------------
# Spec schema
# ---------------------------
# spec supports:
#  - "_bin": executable name/path (default depends on run_ffmpeg_spec/run_ffprobe_spec)
#  - "_inputs": list of inputs
#       each input:
#         - "path": str|Path
#         - "options": dict of input options (appear BEFORE -i)
#       or a plain str|Path input path (no input options)
#  - "_outputs": list of outputs
#       each output:
#         - "path": str|Path
#         - "options": dict of output options (appear BEFORE output path)
#       or a plain str|Path output path (no output options)
#  - "_global": dict of global options (appear right after executable)
#  - "_flags": raw argv list appended at end (escape hatch)
#
# option dict items:
#  - key: str (without leading '-' or with leading '-'; both accepted)
#  - value:
#      - True  => '-key'
#      - False/None => omitted
#      - scalar => '-key value'
#      - list/tuple => repeat: '-key v1 -key v2 ...'


def _opt_key(k: str) -> str:
    return k if k.startswith("-") else f"-{k}"

# 将 python 字典转换为 命令行参数列表
def _opts_to_argv(opts: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for k, v in opts.items():
        kk = _opt_key(k)
        if v is None or v is False:
            continue
        if v is True:
            argv.append(kk)
            continue
        if isinstance(v, (list, tuple)):
            for item in v:
                if item is None or item is False:
                    continue
                if item is True:
                    argv.append(kk)
                else:
                    argv.extend([kk, str(item)])
            continue
        argv.extend([kk, str(v)])
    return argv

# 把 FFmpeg 晦涩的“顺序敏感型”命令行语法，抽象成逻辑清晰的树状结构
def build_argv(spec: Dict[str, Any], *, default_bin: str) -> List[str]:
    """
    Build argv from spec.
    This is order-sensitive and intentionally structured:
      bin + global opts + (input opts + -i path)* + (output opts + out_path)* + flags
    """
    bin_name = spec.get("_bin") or default_bin
    argv: List[str] = [str(bin_name)]

    global_opts = spec.get("_global") or {}
    argv.extend(_opts_to_argv(global_opts))

    for inp in spec.get("_inputs", []) or []:
        if isinstance(inp, (str, Path)):
            argv.extend(["-i", str(inp)])
        else:
            in_opts = inp.get("options") or {}
            argv.extend(_opts_to_argv(in_opts))
            argv.extend(["-i", str(inp["path"])])

    for out in spec.get("_outputs", []) or []:
        if isinstance(out, (str, Path)):
            argv.append(str(out))
        else:
            out_opts = out.get("options") or {}
            argv.extend(_opts_to_argv(out_opts))
            argv.append(str(out["path"]))

    flags = spec.get("_flags") or []
    argv.extend([str(x) for x in flags])

    return argv

# 跨越 Python，调用操作系统底层的进程
def run_cmd(argv: List[str], *, timeout: Optional[float] = None) -> CmdResult:
    """
    Run command without shell (cross-platform safe).
    Captures stdout/stderr for diagnosis.
    """
    try:
        p = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return CmdResult(
            ok=(p.returncode == 0),
            returncode=p.returncode,
            cmd=argv,
            stdout=p.stdout or "",
            stderr=p.stderr or "",
        )
    except FileNotFoundError as e:
        # executable not found
        return CmdResult(
            ok=False,
            returncode=127,
            cmd=argv,
            stdout="",
            stderr=f"Executable not found: {argv[0]!r}. "
                   f"Check PATH or set env var FFMPEG_BIN/FFPROBE_BIN. ({e})",
        )


def run_ffmpeg_spec(spec: Dict[str, Any], *, timeout: Optional[float] = None) -> CmdResult:
    argv = build_argv(spec, default_bin=get_ffmpeg_bin())
    return run_cmd(argv, timeout=timeout)


def run_ffprobe_spec(spec: Dict[str, Any], *, timeout: Optional[float] = None) -> CmdResult:
    argv = build_argv(spec, default_bin=get_ffprobe_bin())
    return run_cmd(argv, timeout=timeout)

# 在耗费时间/算力前检查 ffmpeg/ffprobe 是否可用
def ensure_tools_available() -> None:
    """
    Optional helper: raise with a clear message if ffmpeg/ffprobe not found.
    """
    ffmpeg = get_ffmpeg_bin()
    ffprobe = get_ffprobe_bin()
    if not shutil.which(ffmpeg) and not Path(ffmpeg).exists():
        raise RuntimeError(
            f"ffmpeg not found. Install ffmpeg or set env FFMPEG_BIN. resolved={ffmpeg!r}"
        )
    if not shutil.which(ffprobe) and not Path(ffprobe).exists():
        raise RuntimeError(
            f"ffprobe not found. Install ffprobe or set env FFPROBE_BIN. resolved={ffprobe!r}"
        )
