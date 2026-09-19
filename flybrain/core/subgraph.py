"""Extract a renumbered CSR subgraph. Used for tests that must run on both
devices without a 4 GB allocation."""
from __future__ import annotations

import numpy as np


def extract_subgraph(graph, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.asarray(keep)
    n = len(keep)
    remap = np.full(graph.n_neurons, -1, dtype=np.int64)
    remap[keep] = np.arange(n)

    out_indptr = np.zeros(n + 1, dtype=np.int64)
    out_indices: list[np.ndarray] = []
    out_weights: list[np.ndarray] = []

    for new_i, old_i in enumerate(keep):
        lo, hi = graph.indptr[old_i], graph.indptr[old_i + 1]
        targets = remap[graph.indices[lo:hi]]
        mask = targets >= 0
        out_indices.append(targets[mask])
        out_weights.append(graph.weights[lo:hi][mask])
        out_indptr[new_i + 1] = out_indptr[new_i] + int(mask.sum())

    indices = (
        np.concatenate(out_indices).astype(np.int32)
        if out_indices
        else np.zeros(0, np.int32)
    )
    weights = (
        np.concatenate(out_weights).astype(np.float32)
        if out_weights
        else np.zeros(0, np.float32)
    )
    return out_indptr, indices, weights
