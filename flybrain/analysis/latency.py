"""Encoder-free engine diagnostics.

Inject current directly into an identified population and measure how its
connectome-predicted partners respond. No visual pathway is involved, so a
failure here is unambiguously a simulation fault.

`inject_and_record` also reports the source population's own first-spike
time. `first_spike_ms` is measured from injection onset, which bundles the
source's own integrate-to-threshold time with the actual synaptic delay;
callers that want a genuine monosynaptic latency (presynaptic spike to
postsynaptic spike) must subtract `source_first_spike_ms` themselves.
"""
from __future__ import annotations

import numpy as np
import torch


def monosynaptic_targets(graph, source_idx: np.ndarray, min_weight: float = 5.0) -> np.ndarray:
    """Distinct postsynaptic partners reached by an edge of at least the given
    absolute weight."""
    collected: list[np.ndarray] = []
    for s in np.asarray(source_idx):
        lo, hi = graph.indptr[s], graph.indptr[s + 1]
        seg_w = np.abs(graph.weights[lo:hi])
        collected.append(graph.indices[lo:hi][seg_w >= min_weight])
    if not collected:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(collected)).astype(np.int64)


def inject_and_record(
    net,
    source_idx: np.ndarray,
    watch_idx: np.ndarray,
    amplitude: float = 30.0,
    steps: int = 60,
    inject_steps: int = 5,
) -> dict:
    net.reset()
    source = torch.as_tensor(np.asarray(source_idx), dtype=torch.long, device=net.device)
    watch = torch.as_tensor(np.asarray(watch_idx), dtype=torch.long, device=net.device)

    # v/i_syn are float64; drive tensors must match i_syn's dtype so the
    # add_ inside step() is not silently up-casting a float32 current.
    drive = torch.zeros(net.n_neurons, dtype=net.i_syn.dtype, device=net.device)
    drive[source] = amplitude
    silent = torch.zeros_like(drive)

    first = np.full(len(watch_idx), np.nan, dtype=np.float64)
    raster = np.zeros((steps, len(watch_idx)), dtype=bool)
    source_first = np.nan

    for t in range(steps):
        spikes = net.step(drive if t < inject_steps else silent)
        watched = spikes[watch].cpu().numpy()
        raster[t] = watched
        newly = watched & np.isnan(first)
        first[newly] = t * net.params.dt

        if np.isnan(source_first) and bool(spikes[source].any()):
            source_first = t * net.params.dt

    return {
        "first_spike_ms": first,
        "raster": raster,
        "n_responding": int(np.sum(~np.isnan(first))),
        "source_first_spike_ms": source_first,
    }
