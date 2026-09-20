import numpy as np
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.tuning import homeostatic_tune


def chain_net(n=60, weight=20.0):
    """A ring so activity can sustain itself and gain actually matters.

    weight=20.0 (not the brief's 1.2): with LIFParams() defaults, a constant
    drive of 1.5/step already pushes v to steady state -43.7 mV, above
    v_th=-45 mV, by drive alone -- so at weight=1.2 the observed rate is a
    pure function of the external drive and is invariant to net.gain over
    the whole downward-tuning range (verified empirically: gain in
    [1e-3, 5] all give exactly 23.33 Hz). That makes "moves toward target"
    unfalsifiable for this network. Raising the recurrent weight makes the
    gain-scaled recurrent contribution comparable to the drive-only steady
    current, so reducing gain measurably reduces the rate, which is what
    this test is actually meant to exercise.
    """
    indptr = np.arange(n + 1, dtype=np.int64)
    indices = ((np.arange(n) + 1) % n).astype(np.int32)
    weights = np.full(n, weight, dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams())


def test_tuning_moves_rates_toward_target():
    net = chain_net()
    groups = {"ring": np.arange(net.n_neurons)}

    def drive(_):
        return torch.full((net.n_neurons,), 1.5)

    before = _measure(net, drive)
    homeostatic_tune(net, groups, drive, target_hz=5.0, rounds=20, steps_per_round=300)
    after = _measure(net, drive)
    assert abs(after - 5.0) < abs(before - 5.0), (
        f"tuning moved rate from {before:.2f} Hz to {after:.2f} Hz, away from 5 Hz"
    )


def test_tuning_returns_a_gain_per_group():
    net = chain_net()
    groups = {"a": np.arange(0, 30), "b": np.arange(30, 60)}
    gains = homeostatic_tune(
        net, groups, lambda _: torch.full((60,), 1.5), rounds=3, steps_per_round=100
    )
    assert set(gains) == {"a", "b"}
    assert all(g > 0 for g in gains.values())


def test_gains_are_applied_to_the_network():
    net = chain_net()
    groups = {"all": np.arange(net.n_neurons)}
    gains = homeostatic_tune(
        net, groups, lambda _: torch.full((60,), 1.5), rounds=3, steps_per_round=100
    )
    assert abs(float(net.gain[0]) - gains["all"]) < 1e-6


def test_tuning_is_deterministic_for_identical_inputs():
    """Real and null graphs must be tuned by an identical, reproducible
    procedure or the comparison means nothing."""
    results = []
    for _ in range(2):
        net = chain_net()
        results.append(
            homeostatic_tune(
                net,
                {"all": np.arange(60)},
                lambda _: torch.full((60,), 1.5),
                rounds=5,
                steps_per_round=200,
            )["all"]
        )
    assert abs(results[0] - results[1]) < 1e-9


def _measure(net, drive, steps=300):
    net.reset()
    count = 0
    for t in range(steps):
        count += int(net.step(drive(t)).sum())
    return count / net.n_neurons / (steps * net.params.dt / 1000.0)
