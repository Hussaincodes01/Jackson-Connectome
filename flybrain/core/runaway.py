"""Detect sustained saturation.

Live and batch fail differently on purpose: a frozen stream is dead, but an
experiment that silently clamps its gain and keeps recording produces
contaminated data.
"""
from __future__ import annotations


class RunawayHalt(RuntimeError):
    """Raised by the batch driver when activity saturates."""


class RunawayDetector:
    def __init__(self, ceiling: float = 0.15, window: int = 50, mode: str = "clamp") -> None:
        if mode not in ("clamp", "halt"):
            raise ValueError(f"mode must be 'clamp' or 'halt', got {mode!r}")
        self.ceiling = ceiling
        self.window = window
        self.mode = mode
        self._consecutive = 0
        self.tripped = False

    def update(self, spike_fraction: float) -> str:
        if spike_fraction > self.ceiling:
            self._consecutive += 1
        else:
            self._consecutive = 0
        if self._consecutive >= self.window:
            self.tripped = True
            self._consecutive = 0
            return self.mode
        return "ok"
