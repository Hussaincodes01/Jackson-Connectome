"""Synthetic stimuli with published response properties to compare against."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def drifting_grating(
    height: int,
    width: int,
    direction_deg: float,
    spatial_period_px: float = 24.0,
    speed_px_per_frame: float = 2.0,
) -> Callable[[int], np.ndarray]:
    theta = math.radians(direction_deg)
    ys, xs = np.mgrid[0:height, 0:width]
    projection = xs * math.cos(theta) + ys * math.sin(theta)

    def generate(index: int) -> np.ndarray:
        phase = 2 * math.pi * (projection - index * speed_px_per_frame) / spatial_period_px
        # cos, not sin: for a pure sinusoid, direction 0 vs 180 is a mirror
        # flip of the same travelling wave (projection negates), and at
        # index * speed_px_per_frame == spatial_period_px / 4 a sine-based
        # phase collapses to an identical frame for every pixel (cos of the
        # offset term hits exactly zero). cos hits that degeneracy at a
        # different, non-integer-frame offset, so opposite directions stay
        # distinguishable at the frame indices stimuli sweeps actually use.
        return ((np.cos(phase) + 1.0) / 2.0).astype(np.float32)

    return generate


def looming_disc(
    height: int,
    width: int,
    r_over_v: float = 40.0,
    n_frames: int = 90,
) -> Callable[[int], np.ndarray]:
    """A dark disc expanding from the centre -- the classic escape stimulus.

    Angular size follows theta(t) = 2 * arctan(r / (v * t_to_collision)), so
    growth accelerates as collision approaches, as it does for a real
    approaching object.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    distance = np.sqrt((xs - width / 2.0) ** 2 + (ys - height / 2.0) ** 2)
    max_radius = min(height, width) / 2.0

    def generate(index: int) -> np.ndarray:
        remaining = max(n_frames - index, 1)
        theta = 2.0 * math.atan(r_over_v / remaining)
        radius = min(theta / (math.pi / 2.0), 1.0) * max_radius
        frame = np.ones((height, width), dtype=np.float32)
        frame[distance <= radius] = 0.0
        return frame

    return generate


def contrast_step(
    height: int,
    width: int,
    low: float = 0.2,
    high: float = 0.8,
    switch_frame: int = 30,
) -> Callable[[int], np.ndarray]:
    def generate(index: int) -> np.ndarray:
        value = high if index >= switch_frame else low
        return np.full((height, width), value, dtype=np.float32)

    return generate
