# QfinVidCore/utils/runtime/system.py
"""System-level runtime helpers."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from typing import Optional, Sequence, Union


def which(program: str) -> Optional[str]:
    """Locate an executable in PATH."""
    return shutil.which(program)


def run_cmd(
    args: Union[str, Sequence[str]],
    check: bool = False,
    capture_output: bool = False,
    text: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess."""
    argv = shlex.split(args) if isinstance(args, str) else list(args)
    return subprocess.run(
        argv,
        check=check,
        capture_output=capture_output,
        text=text,
        **kwargs,
    )


def get_program_version(program: str) -> str:
    """Return program version summary or status string."""
    try:
        cp = run_cmd([program, "-version"], capture_output=True, text=True)
        if cp.returncode != 0:
            return "unknown"
        line = (cp.stdout or "").splitlines()
        return line[0] if line else "unknown"
    except Exception:
        return "not_installed"


def get_git_version() -> str:
    """Return current git short hash or status string."""
    try:
        cp = run_cmd(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        if cp.returncode == 0:
            return (cp.stdout or "").strip()
        return "unknown"
    except Exception:
        return "unknown"
