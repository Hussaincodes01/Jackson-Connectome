"""Resolve presynaptic neurotransmitter identity to a synaptic sign."""
from pathlib import Path

import numpy as np
import yaml

UNKNOWN = {None, "unclear", ""}


def load_sign_map(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return {str(k): int(v) for k, v in raw.items()}


def resolve_signs(
    consensus_nt: np.ndarray,
    celltype_nt: np.ndarray,
    sign_map: dict[str, int],
) -> np.ndarray:
    """Per-neuron sign. Falls back to cell-type consensus, then 0.

    Sign 0 covers modulators (dopamine/octopamine/serotonin) and genuinely
    unknown neurons alike -- both contribute no output under a two-sign LIF
    model. The build stage drops their out-edges and tallies them separately.
    """
    n = len(consensus_nt)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        primary = consensus_nt[i]
        if primary in UNKNOWN:
            primary = celltype_nt[i]
        if primary in UNKNOWN:
            continue
        out[i] = sign_map.get(str(primary), 0)
    return out
