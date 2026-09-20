"""Map the optic lobe's hexagonal column lattice onto image pixels.

892 columns tile each lobe. Coverage is not perfect -- some columns have a
neuron on only one side -- so the builder takes what exists rather than
assuming a complete tiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class HexLattice:
    neuron_idx: np.ndarray
    hex1: np.ndarray
    hex2: np.ndarray
    xy: np.ndarray
    side: str


def build_lattice(graph, cell_type: str, side: str) -> HexLattice:
    candidates = graph.type_index(cell_type)
    on_side = candidates[graph.soma_side[candidates] == side]
    has_coords = on_side[
        ~np.isnan(graph.hex1[on_side]) & ~np.isnan(graph.hex2[on_side])
    ]

    h1 = graph.hex1[has_coords].astype(np.float64)
    h2 = graph.hex2[has_coords].astype(np.float64)

    x = h1 + h2 / 2.0
    y = h2 * (math.sqrt(3.0) / 2.0)

    def norm(a: np.ndarray) -> np.ndarray:
        span = a.max() - a.min()
        return (a - a.min()) / span if span > 0 else np.zeros_like(a)

    return HexLattice(
        neuron_idx=has_coords,
        hex1=h1,
        hex2=h2,
        xy=np.stack([norm(x), norm(y)], axis=1),
        side=side,
    )


def sampling_matrix(
    lattice: HexLattice,
    height: int,
    width: int,
    sigma_px: float = 3.0,
    fov_fraction: float = 0.9,
) -> sp.csr_matrix:
    """Each column draws a Gaussian-weighted patch. Rows sum to 1, so a
    uniform frame yields a uniform column response.

    `fov_fraction` insets the lattice from the frame edge -- the fly's real
    field of view is far wider than a webcam's, so the lattice occupies the
    central region and the periphery is simply dark.
    """
    margin = (1.0 - fov_fraction) / 2.0
    cx = (margin + lattice.xy[:, 0] * fov_fraction) * (width - 1)
    cy = (margin + lattice.xy[:, 1] * fov_fraction) * (height - 1)

    radius = int(math.ceil(3 * sigma_px))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for c in range(len(cx)):
        x0, x1 = max(0, int(cx[c]) - radius), min(width, int(cx[c]) + radius + 1)
        y0, y1 = max(0, int(cy[c]) - radius), min(height, int(cy[c]) + radius + 1)
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        gx = np.exp(-((xs - cx[c]) ** 2) / (2 * sigma_px**2))
        gy = np.exp(-((ys - cy[c]) ** 2) / (2 * sigma_px**2))
        patch = np.outer(gy, gx)
        total = patch.sum()
        if total <= 0:
            continue
        patch /= total
        flat = (ys[:, None] * width + xs[None, :]).ravel()
        rows.append(np.full(flat.size, c, dtype=np.int32))
        cols.append(flat.astype(np.int32))
        vals.append(patch.ravel().astype(np.float32))

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(cx), height * width),
    )
