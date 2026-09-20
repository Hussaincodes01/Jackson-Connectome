import numpy as np
import pytest
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.runaway import RunawayDetector, RunawayHalt
from flybrain.drivers.batch import run_batch
from flybrain.encode.source import Frame, SyntheticSource


class StubEncoder:
    """Drives neuron 0 in proportion to mean frame luminance."""

    def __init__(self, n):
        self.n = n

    def reset(self):
        pass

    def encode(self, frame: Frame) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        out[0] = float(frame.image.mean()) * 40.0
        return out


def tiny_net(n=4):
    indptr = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    indices = np.array([1], dtype=np.int32)
    weights = np.array([9.0], dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams())


def test_returns_one_value_per_frame_per_group():
    net = tiny_net()
    src = SyntheticSource(lambda i: np.full((4, 4), 1.0, np.float32), 4, 4)
    out = run_batch(
        net, StubEncoder(4), src, {"driven": np.array([0]), "target": np.array([1])},
        n_frames=5, steps_per_frame=10,
    )
    assert out["driven"].shape == (5,)
    assert out["target"].shape == (5,)
    assert out["spike_fraction"].shape == (5,)


def test_bright_input_produces_spikes_in_the_driven_group():
    net = tiny_net()
    src = SyntheticSource(lambda i: np.full((4, 4), 1.0, np.float32), 4, 4)
    out = run_batch(
        net, StubEncoder(4), src, {"driven": np.array([0])},
        n_frames=5, steps_per_frame=10,
    )
    assert out["driven"].sum() > 0


def test_dark_input_produces_no_spikes():
    net = tiny_net()
    src = SyntheticSource(lambda i: np.zeros((4, 4), np.float32), 4, 4)
    out = run_batch(
        net, StubEncoder(4), src, {"driven": np.array([0])},
        n_frames=5, steps_per_frame=10,
    )
    assert out["driven"].sum() == 0


def test_batch_driver_halts_on_runaway():
    """Batch must halt, never silently clamp -- clamped data is contaminated."""
    net = tiny_net()
    src = SyntheticSource(lambda i: np.full((4, 4), 1.0, np.float32), 4, 4)
    detector = RunawayDetector(ceiling=0.0, window=1, mode="halt")
    with pytest.raises(RunawayHalt):
        run_batch(
            net, StubEncoder(4), src, {"driven": np.array([0])},
            n_frames=5, steps_per_frame=10, detector=detector,
        )
