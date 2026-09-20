import numpy as np
import pytest

from flybrain.analysis.latency import inject_and_record, monosynaptic_targets
from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph():
    return load_graph(ARTIFACTS / "graph_t2.npz")


@pytest.fixture(scope="module")
def net(graph):
    return LIFNetwork.from_graph(graph, LIFParams(), device="cuda")


def test_lplc2_has_monosynaptic_targets(graph):
    src = graph.type_index("LPLC2")
    assert len(src) == 185
    targets = monosynaptic_targets(graph, src, min_weight=5.0)
    assert len(targets) > 0, "LPLC2 has no strong outputs -- ingest is wrong"


def test_driven_population_actually_spikes(net, graph):
    src = graph.type_index("LPLC2")
    out = inject_and_record(net, src, src, amplitude=30.0, steps=40)
    assert out["n_responding"] > 0, "injected population did not spike"


def test_monosynaptic_latency_is_one_to_three_ms(net, graph):
    """One synapse costs one timestep of transmission plus membrane charging,
    measured from the source's OWN spike time -- not from injection onset,
    which would also bundle in how long the source takes to reach threshold.
    Anything outside 1-3 ms means the propagation path is wrong."""
    src = graph.type_index("LPLC2")
    targets = np.setdiff1d(monosynaptic_targets(graph, src, min_weight=10.0), src)
    assert len(targets) >= 5, "not enough strong postsynaptic partners to test"

    out = inject_and_record(net, src, targets, amplitude=30.0, steps=60)
    assert not np.isnan(out["source_first_spike_ms"]), "source population never fired"

    relative = out["first_spike_ms"] - out["source_first_spike_ms"]
    responded = relative[~np.isnan(relative)]
    assert len(responded) > 0, "no monosynaptic partner responded"

    median = float(np.median(responded))
    assert 1.0 <= median <= 3.0, (
        f"median monosynaptic latency {median} ms is out of range "
        f"(source fired at {out['source_first_spike_ms']} ms)"
    )


def test_partners_that_respond_are_connectome_partners(net, graph):
    """Responses must land on neurons the wiring actually predicts, not
    scattered across the network."""
    src = graph.type_index("LPLC2")
    targets = np.setdiff1d(monosynaptic_targets(graph, src, min_weight=10.0), src)
    unconnected = np.setdiff1d(
        graph.type_index("L1"), monosynaptic_targets(graph, src, min_weight=0.0)
    )[: len(targets)]

    connected_out = inject_and_record(net, src, targets, amplitude=30.0, steps=60)
    control_out = inject_and_record(net, src, unconnected, amplitude=30.0, steps=60)

    connected_rate = np.mean(~np.isnan(connected_out["first_spike_ms"]))
    control_rate = np.mean(~np.isnan(control_out["first_spike_ms"]))
    assert connected_rate > control_rate, (
        f"connected partners responded at {connected_rate:.2%} but unconnected "
        f"neurons at {control_rate:.2%} -- propagation is not following the wiring"
    )


def test_inhibitory_source_suppresses_rather_than_drives(net, graph):
    """A GABAergic population must lower its targets' voltages."""
    # Only rows that actually have out-edges; indptr[i] == indptr[i+1] for an
    # empty row and would index past the end of weights for trailing rows.
    has_edges = np.flatnonzero(np.diff(graph.indptr) > 0)
    starts = graph.indptr[has_edges]
    gaba_sources = has_edges[graph.weights[starts] < 0]
    assert len(gaba_sources) > 0
    src = gaba_sources[:200]
    targets = monosynaptic_targets(graph, src, min_weight=0.0)[:500]
    assert len(targets) > 0

    net.reset()
    baseline = float(net.v[targets].mean())
    inject_and_record(net, src, targets, amplitude=30.0, steps=30)
    assert float(net.v[targets].mean()) <= baseline + 1e-3
