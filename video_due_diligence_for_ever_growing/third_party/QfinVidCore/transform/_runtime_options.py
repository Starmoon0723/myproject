# QfinVidCore/transform/_runtime_options.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(slots=True)
class RunOptions:
    """Shared ffmpeg execution options for transform modules."""

    timeout: Optional[float] = None
    input_options: Mapping[str, Any] = field(default_factory=dict)
    output_options: Mapping[str, Any] = field(default_factory=dict)
    global_options: Mapping[str, Any] = field(default_factory=dict)
    flags: Sequence[Any] = field(default_factory=tuple)
