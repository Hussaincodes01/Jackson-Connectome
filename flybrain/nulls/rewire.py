"""Shuffled-wiring controls.

Every model preserves Dale's law, because the real graph does. A null that
broke it would differ from the real network in two ways at once, and the
comparison would no longer isolate the contribution of wiring.
"""
from __future__ import annotations

import numpy as np


def _row_signs(indptr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """One sign per neuron, taken from its first out-edge (Dale's law)."""
    n = len(indptr) - 1
    signs = np.zeros(n, dtype=np.float32)
    has_edges = np.diff(indptr) > 0
    rows = np.flatnonzero(has_edges)
    signs[rows] = np.sign(weights[indptr[rows]])
    return signs


def _expand_row_signs(indptr: np.ndarray, signs: np.ndarray, n_edges: int) -> np.ndarray:
    counts = np.diff(indptr)
    return np.repeat(signs, counts)[:n_edges]


def weight_permutation(indptr, indices, weights, seed: int):
    """Shuffle magnitudes; keep topology and each edge's sign."""
    rng = np.random.default_rng(seed)
    magnitudes = np.abs(weights).copy()
    rng.shuffle(magnitudes)
    signed = magnitudes * np.sign(weights)
    return indptr.copy(), indices.copy(), signed.astype(np.float32)


def degree_preserving_rewire(indptr, indices, weights, seed: int):
    """Permute the entire target list.

    Out-degree is preserved because row lengths are untouched; in-degree is
    preserved exactly because the multiset of targets is unchanged.
    """
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return indptr.copy(), shuffled, weights.copy()


def sign_permutation(indptr, indices, weights, seed: int):
    """Permute which neurons are excitatory and which inhibitory."""
    rng = np.random.default_rng(seed)
    signs = _row_signs(indptr, weights)
    active = np.flatnonzero(signs != 0)
    permuted = signs.copy()
    permuted[active] = rng.permutation(signs[active])
    edge_old = _expand_row_signs(indptr, signs, len(weights))
    edge_new = _expand_row_signs(indptr, permuted, len(weights))
    out = np.abs(weights) * edge_new
    # Edges from sign-0 rows cannot occur in a compiled graph, but stay safe.
    out = np.where(edge_new == 0, weights * 0.0, out)
    del edge_old
    return indptr.copy(), indices.copy(), out.astype(np.float32)


def type_preserving_rewire(indptr, indices, weights, types: np.ndarray, seed: int):
    """Randomise individual partners while holding cell-type-to-cell-type
    connection counts fixed. The strictest null: it asks whether the specific
    wiring matters beyond the type-level summary."""
    rng = np.random.default_rng(seed)
    counts = np.diff(indptr)
    source_of_edge = np.repeat(np.arange(len(counts)), counts)
    pre_types = types[source_of_edge]
    post_types = types[indices]

    new_indices = indices.copy()
    key = np.char.add(np.char.add(pre_types.astype("<U32"), "->"), post_types.astype("<U32"))
    for block in np.unique(key):
        members = np.flatnonzero(key == block)
        if members.size > 1:
            new_indices[members] = rng.permutation(indices[members])
    return indptr.copy(), new_indices, weights.copy()


NULL_MODELS = {
    "weight_permutation": weight_permutation,
    "degree_preserving": degree_preserving_rewire,
    "sign_permutation": sign_permutation,
    "type_preserving": type_preserving_rewire,
}
