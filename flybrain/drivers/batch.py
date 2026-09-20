"""Stimulus -> recorded activity.

Halts on runaway rather than clamping: an experiment that silently reduces its
gain and keeps recording produces data that looks fine and is not.
"""
from __future__ import annotations

import numpy as np
import torch

from flybrain.core.runaway import RunawayHalt


def run_batch(
    net,
    encoder,
    source,
    watch: dict[str, np.ndarray],
    n_frames: int,
    steps_per_frame: int = 33,
    detector=None,
    burn_in_frames: int = 1,
) -> dict[str, np.ndarray]:
    """Run a stimulus sweep and record per-frame spike counts.

    `burn_in_frames` frames are run BEFORE recording starts and are not
    included in the results. This is not cosmetic: the encoder's running
    luminance mean is empty immediately after `reset()`, so the first frame's
    contrast is computed against a dark-adapted (zero) baseline and comes out
    fully saturated at the clip bound, every time. Recording it would put an
    identical, stimulus-independent spike of activity at index 0 of every
    condition -- real and null alike -- which is exactly the kind of shared
    artifact that flatters a comparison. It resolves on the very next frame,
    because the running mean is seeded directly from frame 0 rather than
    blended in, so one burn-in frame is sufficient.
    """
    if detector is not None and detector.mode != "halt":
        raise ValueError(
            f"run_batch requires a halt-mode detector, got mode={detector.mode!r}. "
            "The batch driver must never clamp and keep recording: an experiment "
            "that silently reduces its gain produces contaminated data that looks "
            "like a clean run. Clamping belongs to the live driver."
        )

    net.reset()
    if hasattr(encoder, "reset"):
        encoder.reset()

    for _ in range(burn_in_frames):
        frame = source.read()
        current = torch.as_tensor(encoder.encode(frame), device=net.device)
        for _ in range(steps_per_frame):
            net.step(current)

    results = {name: np.zeros(n_frames, dtype=np.int64) for name in watch}
    results["spike_fraction"] = np.zeros(n_frames, dtype=np.float64)
    groups = {
        name: torch.as_tensor(np.asarray(idx), dtype=torch.long, device=net.device)
        for name, idx in watch.items()
    }

    for f in range(n_frames):
        frame = source.read()
        current = torch.as_tensor(encoder.encode(frame), device=net.device)
        fractions = []

        for _ in range(steps_per_frame):
            spikes = net.step(current)
            fractions.append(net.last_spike_fraction)
            for name, idx in groups.items():
                results[name][f] += int(spikes[idx].sum())
            if detector is not None:
                action = detector.update(net.last_spike_fraction)
                if action == "halt":
                    raise RunawayHalt(
                        f"activity saturated at frame {f}: "
                        f"{net.last_spike_fraction:.1%} of neurons spiking"
                    )

        results["spike_fraction"][f] = float(np.mean(fractions))

    return results
