"""Event-driven leaky integrate-and-fire over a signed CSR connectome.

This module is pure. It must not import from flybrain.encode, flybrain.readout,
flybrain.drivers or flybrain.bridge -- a test enforces that. It takes an input
current vector, advances one timestep, and exposes a spike mask. Nothing else.

Integration is exponential-Euler, so the input-free solution is exact:
    V(t) = V_rest + (V0 - V_rest) * exp(-t / tau_m)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LIFParams:
    """Millivolts and milliseconds throughout. Defaults follow the
    connectome-LIF literature (Shiu et al., Nature 2024)."""

    v_rest: float = -52.0
    v_th: float = -45.0
    v_reset: float = -52.0
    tau_m: float = 20.0
    tau_syn: float = 5.0
    t_ref: float = 2.2
    dt: float = 1.0


class LIFNetwork:
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        params: LIFParams | None = None,
        device: str = "cpu",
    ) -> None:
        self.params = params or LIFParams()
        self.device = torch.device(device)
        self._n = len(indptr) - 1

        self.indptr = torch.as_tensor(np.asarray(indptr), dtype=torch.int64, device=self.device)
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.int64, device=self.device)
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.float32, device=self.device)

        p = self.params
        self.decay_m = math.exp(-p.dt / p.tau_m)
        self.decay_s = math.exp(-p.dt / p.tau_syn) if p.tau_syn > 0 else 0.0
        self.n_ref = int(round(p.t_ref / p.dt))

        self.gain = torch.ones(self._n, dtype=torch.float32, device=self.device)
        self.reset()

    @classmethod
    def from_graph(cls, graph, params: LIFParams | None = None, device: str = "cpu") -> "LIFNetwork":
        return cls(graph.indptr, graph.indices, graph.weights, params, device)

    @property
    def n_neurons(self) -> int:
        return self._n

    def reset(self) -> None:
        p = self.params
        # v and i_syn are float64: the exponential-Euler recurrence is exact
        # in exact arithmetic, so the 1e-6 closed-form tolerance is a
        # precision requirement, not just a scheme requirement -- float32
        # rounding alone exceeds it within a couple of steps at mV scale.
        self.v = torch.full((self._n,), p.v_rest, dtype=torch.float64, device=self.device)
        self.i_syn = torch.zeros(self._n, dtype=torch.float64, device=self.device)
        self.refrac = torch.zeros(self._n, dtype=torch.int16, device=self.device)
        self.last_spike_fraction = 0.0

    def _propagate(self, spike_idx: torch.Tensor) -> None:
        """Gather only the out-edges of neurons that actually fired.

        Dense propagation would touch all 57.7M edges every step and cap the
        simulation near 240 steps/s. This is the hot loop; keep it event-driven.
        """
        if spike_idx.numel() == 0:
            return
        starts = self.indptr[spike_idx]
        counts = self.indptr[spike_idx + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        row_of = torch.repeat_interleave(
            torch.arange(spike_idx.numel(), device=self.device), counts
        )
        offsets = torch.cumsum(counts, 0) - counts
        flat = starts[row_of] + (
            torch.arange(total, device=self.device) - offsets[row_of]
        )
        targets = self.indices[flat]
        contrib = (self.weights[flat] * self.gain[spike_idx][row_of]).to(self.i_syn.dtype)
        self.i_syn.index_add_(0, targets, contrib)

    def step(self, input_current: torch.Tensor | None = None) -> torch.Tensor:
        p = self.params

        self.i_syn.mul_(self.decay_s)
        if input_current is not None:
            self.i_syn.add_(input_current.to(self.device, self.i_syn.dtype))

        # Exponential-Euler membrane update.
        self.v.copy_(
            p.v_rest
            + (self.v - p.v_rest) * self.decay_m
            + self.i_syn * (1.0 - self.decay_m)
        )

        refractory = self.refrac > 0
        if bool(refractory.any()):
            self.v = torch.where(
                refractory, torch.full_like(self.v, p.v_reset), self.v
            )
            self.refrac = torch.clamp(self.refrac - 1, min=0)

        spikes = self.v >= p.v_th
        spike_idx = spikes.nonzero(as_tuple=True)[0]
        if spike_idx.numel():
            self.v[spike_idx] = p.v_reset
            self.refrac[spike_idx] = self.n_ref
            self._propagate(spike_idx)

        self.last_spike_fraction = spike_idx.numel() / max(self._n, 1)
        return spikes
