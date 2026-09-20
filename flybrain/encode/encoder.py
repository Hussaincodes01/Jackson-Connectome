"""Luminance -> adapted contrast -> lamina current.

Input enters at L1/L2 rather than the photoreceptors: photoreceptors carry no
hex-column assignment in this dataset, and the photoreceptor->LMC synapse is
histaminergic and inhibitory, so the sign inversion is already accounted for.

L1 carries the ON pathway and L2 the OFF pathway, so they receive opposite
contrast polarity.
"""
from __future__ import annotations

import math

import numpy as np

from flybrain.encode.lattice import build_lattice, sampling_matrix
from flybrain.encode.source import Frame


class LaminaEncoder:
    def __init__(
        self,
        graph,
        height: int,
        width: int,
        tau_adapt_ms: float = 500.0,
        drive_mv: float = 8.0,
        dt_ms: float = 33.0,
        sigma_px: float = 3.0,
    ) -> None:
        self.n_neurons = graph.n_neurons
        self.drive_mv = drive_mv
        self.adapt_decay = math.exp(-dt_ms / tau_adapt_ms)

        self.channels = []
        # BOTH lamina monopolar types hyperpolarise to a light increment:
        # photoreceptors are histaminergic and inhibitory onto both. The
        # ON/OFF split is NOT imposed here -- it emerges downstream from the
        # connectome's own signs (L1 is inhibitory, L2 excitatory), which is
        # the whole point of driving a real wiring diagram. An earlier version
        # drove L1 with +1 polarity; L1 then fired 55,955 times and clamped
        # Mi1 -- T4's dominant input -- off entirely, so T4 never spiked.
        for cell_type, polarity in (("L1", -1.0), ("L2", -1.0)):
            for side in ("R", "L"):
                lattice = build_lattice(graph, cell_type, side)
                if len(lattice.neuron_idx) == 0:
                    continue
                self.channels.append(
                    {
                        "idx": lattice.neuron_idx,
                        "matrix": sampling_matrix(lattice, height, width, sigma_px=sigma_px),
                        "polarity": polarity,
                    }
                )
        self.reset()

    def reset(self) -> None:
        self._mean = [None] * len(self.channels)

    def encode(self, frame: Frame) -> np.ndarray:
        flat = frame.image.ravel().astype(np.float32)
        out = np.zeros(self.n_neurons, dtype=np.float32)

        for c, channel in enumerate(self.channels):
            luminance = channel["matrix"] @ flat

            if self._mean[c] is None:
                # Freshly reset: no running mean yet. Treat the prior state
                # as dark-adapted (mean 0) so this first frame still yields a
                # real contrast signal, then seed the running mean from this
                # frame for subsequent calls.
                running = np.zeros_like(luminance)
                self._mean[c] = luminance.copy()
            else:
                running = self._mean[c]
                self._mean[c] = running * self.adapt_decay + luminance * (1.0 - self.adapt_decay)

            # Divisive adaptation against the running local mean.
            contrast = (luminance - running) / (running + 0.05)

            out[channel["idx"]] = (
                channel["polarity"] * self.drive_mv * np.clip(contrast, -4.0, 4.0)
            ).astype(np.float32)

        return out
