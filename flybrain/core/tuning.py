"""Homeostatic gain calibration.

The connectome fixes who connects to whom; it does not fix synaptic strength
in millivolts. Gains are set by driving firing rates into a plausible band.

The procedure is deterministic and must be applied identically to null models.
Tuning the real network into working while leaving controls to fail would
invalidate every comparison in the battery.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def homeostatic_tune(
    net,
    type_groups: dict[str, np.ndarray],
    drive_fn: Callable[[int], torch.Tensor],
    target_hz: float = 5.0,
    rounds: int = 20,
    eta: float = 0.5,
    steps_per_round: int = 500,
    floor: float = 1e-3,
    ceiling: float = 1e3,
) -> dict[str, float]:
    gains = {name: 1.0 for name in type_groups}
    seconds = steps_per_round * net.params.dt / 1000.0

    for _ in range(rounds):
        net.reset()
        counts = torch.zeros(net.n_neurons, dtype=torch.float32, device=net.device)
        for t in range(steps_per_round):
            counts += net.step(drive_fn(t)).to(torch.float32)

        for name, idx in type_groups.items():
            members = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=net.device)
            observed = float(counts[members].mean()) / seconds
            if observed <= 0.0:
                gains[name] = min(gains[name] * (1.0 + eta), ceiling)
            else:
                gains[name] = float(
                    np.clip(gains[name] * (target_hz / observed) ** eta, floor, ceiling)
                )
            net.gain[members] = gains[name]

    net.reset()
    return gains
