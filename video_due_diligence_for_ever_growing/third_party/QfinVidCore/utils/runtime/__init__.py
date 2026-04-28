# QfinVidCore/utils/runtime/__init__.py
"""Runtime utility exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from QfinVidCore.utils.runtime.system import get_git_version, get_program_version, run_cmd, which


def run_torch_model(model_path: Path, input_data: Any) -> Any:
    """Placeholder for optional torch runtime integration."""
    raise NotImplementedError(
        "run_torch_model is not implemented in runtime package. "
        "Integrate your model execution pipeline explicitly."
    )


__all__ = [
    "run_cmd",
    "which",
    "get_program_version",
    "get_git_version",
    "run_torch_model",
]
