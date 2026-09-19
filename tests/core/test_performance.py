import time

import numpy as np
import pytest
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow

# Measured ceiling on the target RTX 2050 is ~435 steps/s for a full step; 1000 is not
# achievable on this hardware. 250 steps/s is real time at dt=4ms and is comfortably
# above the 230 steps/s sync-heavy baseline, so it still fails a non-event-driven or
# sync-laden implementation.
REALTIME_FLOOR_STEPS_PER_SEC = 250


@pytest.fixture(scope="module")
def full_net():
    g = load_graph(ARTIFACTS / "graph_t2.npz")
    return LIFNetwork.from_graph(g, LIFParams(), device="cuda"), g


def test_graph_fits_in_vram_budget(full_net):
    _, g = full_net
    megabytes = (g.weights.nbytes + g.indices.nbytes + g.indptr.nbytes) / 1e6
    assert megabytes < 500, f"graph is {megabytes:.0f} MB, over the 4 GB card's budget"


def test_event_driven_loop_clears_realtime(full_net):
    """Dense propagation caps near 240 steps/s on this card. Anything at or
    below that means the event-driven path is not actually being taken."""
    net, g = full_net
    drive = torch.zeros(net.n_neurons, device="cuda")
    seed = torch.randint(0, net.n_neurons, (2000,), device="cuda")
    drive[seed] = 12.0

    for _ in range(50):          # warm up kernels and autotuning
        net.step(drive)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(500):
        net.step(drive)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    steps_per_sec = 500 / elapsed
    assert steps_per_sec > REALTIME_FLOOR_STEPS_PER_SEC, (
        f"{steps_per_sec:.0f} steps/s is below the measured floor; the hot loop is "
        "either not event-driven or is synchronising with the CPU every step"
    )
    print(f"measured: {steps_per_sec:.0f} steps/s")
