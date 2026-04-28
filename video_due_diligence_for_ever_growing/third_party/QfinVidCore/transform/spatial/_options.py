# QfinVidCore/transform/spatial/_options.py

from __future__ import annotations

from dataclasses import dataclass

from QfinVidCore.transform._runtime_options import RunOptions


@dataclass(slots=True)
class ResizeOptions(RunOptions):
    width: int | None = None
    height: int | None = None
    overwrite: bool | None = None
