"""Effect sizes and confidence intervals for the validation battery."""
from __future__ import annotations

import numpy as np


def direction_selectivity_index(preferred: float, opposite: float) -> float:
    total = preferred + opposite
    if total == 0:
        return 0.0
    return float((preferred - opposite) / total)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def looming_discrimination(loom_counts: np.ndarray, control_counts: np.ndarray) -> float:
    """How much more the escape pathway fires to a looming disc than to a
    non-looming control, normalised to [-1, 1]."""
    loom_total = float(np.sum(loom_counts))
    control_total = float(np.sum(control_counts))
    total = loom_total + control_total
    if total == 0:
        return 0.0
    return (loom_total - control_total) / total
