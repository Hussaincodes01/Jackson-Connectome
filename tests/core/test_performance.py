import time

import numpy as np
import pytest
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow

# Absolute throughput cannot distinguish a good implementation from a bad one on
# this machine: the sync-laden baseline measured 214-230 steps/s and the fixed
# version 231-285, and those ranges OVERLAP. Any floor between them is flaky by
# construction -- best-of-5 was tried and still failed 1 run in 3. So this file
# tests the defining property directly instead of timing it in absolute terms.
#
# CATASTROPHIC_FLOOR catches only gross regression (an accidental dense rewrite, a
# CPU fallback). It sits far below measured capability and is NOT a real-time
# claim: real time at dt=1ms needs 1000 steps/s, which this hardware never reaches.
CATASTROPHIC_FLOOR_STEPS_PER_SEC = 150

# The real property. An event-driven loop's cost scales with how many neurons FIRED,
# not with how many edges exist, so a quiet network must be markedly faster than a
# busy one. A dense implementation touches all 14.8M edges either way and scores
# near 1.0. Measured on the target hardware: 935 steps/s quiet vs 240 busy = 3.89x.
MIN_ACTIVITY_SCALING_RATIO = 2.0


@pytest.fixture(scope="module")
def full_net():
    g = load_graph(ARTIFACTS / "graph_t2.npz")
    return LIFNetwork.from_graph(g, LIFParams(), device="cuda"), g


def test_graph_fits_in_vram_budget(full_net):
    _, g = full_net
    megabytes = (g.weights.nbytes + g.indices.nbytes + g.indptr.nbytes) / 1e6
    assert megabytes < 500, f"graph is {megabytes:.0f} MB, over the 4 GB card's budget"


def _best_steps_per_sec(net, drive, reps=4, steps=200):
    """Best of several repetitions. Boost clocks and any other GPU user make a
    single run a measurement of ambient load as much as of the code."""
    best = 0.0
    for _ in range(reps):
        net.reset()
        for _ in range(30):
            net.step(drive)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            net.step(drive)
        torch.cuda.synchronize()
        best = max(best, steps / (time.perf_counter() - start))
    return best


def test_cost_scales_with_activity_not_edge_count(full_net):
    """The defining property of an event-driven loop, tested as a RATIO so it is
    independent of clock speed and machine load.

    A quiet network gathers almost no out-edges and runs several times faster than
    a busy one. A dense implementation touches all 14.8M edges regardless and would
    score near 1.0."""
    net, _ = full_net
    quiet = torch.zeros(net.n_neurons, device="cuda", dtype=net.i_syn.dtype)

    busy = torch.zeros(net.n_neurons, device="cuda", dtype=net.i_syn.dtype)
    torch.manual_seed(1)
    busy[torch.randint(0, net.n_neurons, (6000,), device="cuda")] = 14.0

    quiet_sps = _best_steps_per_sec(net, quiet)
    busy_sps = _best_steps_per_sec(net, busy)
    ratio = quiet_sps / busy_sps
    print(f"quiet {quiet_sps:.0f} steps/s, busy {busy_sps:.0f} steps/s, ratio {ratio:.2f}x")

    assert ratio > MIN_ACTIVITY_SCALING_RATIO, (
        f"cost barely changed with activity (ratio {ratio:.2f}x); the hot loop is "
        "touching every edge regardless of who fired, i.e. it is not event-driven"
    )


def test_throughput_has_not_catastrophically_regressed(full_net):
    """A floor far below measured capability -- not a real-time claim. See the
    comment on CATASTROPHIC_FLOOR_STEPS_PER_SEC."""
    net, _ = full_net
    drive = torch.zeros(net.n_neurons, device="cuda", dtype=net.i_syn.dtype)
    torch.manual_seed(0)
    drive[torch.randint(0, net.n_neurons, (2000,), device="cuda")] = 12.0

    steps_per_sec = _best_steps_per_sec(net, drive)
    print(f"measured: {steps_per_sec:.0f} steps/s")
    assert steps_per_sec > CATASTROPHIC_FLOOR_STEPS_PER_SEC, (
        f"{steps_per_sec:.0f} steps/s is far below the measured 240-285 range"
    )
