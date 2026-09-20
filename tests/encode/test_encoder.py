import numpy as np
import pytest

from flybrain.encode.encoder import LaminaEncoder
from flybrain.encode.source import Frame
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow

H, W = 120, 160


@pytest.fixture(scope="module")
def graph():
    return load_graph(ARTIFACTS / "graph_t2.npz")


@pytest.fixture(scope="module")
def encoder(graph):
    return LaminaEncoder(graph, H, W)


def frame(value):
    return Frame(image=np.full((H, W), value, np.float32), timestamp=0.0, stale=False)


def test_output_length_matches_neuron_count(encoder, graph):
    out = encoder.encode(frame(0.5))
    assert out.shape == (graph.n_neurons,)
    assert out.dtype == np.float32


def test_only_lamina_neurons_receive_current(encoder, graph):
    encoder.reset()
    out = encoder.encode(frame(1.0))
    driven = np.flatnonzero(out != 0)
    lamina = np.concatenate([graph.type_index("L1"), graph.type_index("L2")])
    assert len(np.setdiff1d(driven, lamina)) == 0, "current leaked outside L1/L2"


def test_l1_and_l2_receive_opposite_polarity(encoder, graph):
    encoder.reset()
    out = encoder.encode(frame(1.0))     # bright against a dark-adapted mean
    l1 = out[graph.type_index("L1")]
    l2 = out[graph.type_index("L2")]
    l1_active = l1[l1 != 0]
    l2_active = l2[l2 != 0]
    assert l1_active.mean() > 0, "L1 should depolarise to positive contrast"
    assert l2_active.mean() < 0, "L2 should carry the opposite sign"


def test_adaptation_decays_response_to_a_constant_scene(encoder):
    """Real photoreceptors adapt. Without this a bright scene saturates the
    model permanently."""
    encoder.reset()
    first = np.abs(encoder.encode(frame(1.0))).sum()
    for _ in range(200):
        encoder.encode(frame(1.0))
    settled = np.abs(encoder.encode(frame(1.0))).sum()
    assert settled < first * 0.5, (
        f"response only fell from {first:.1f} to {settled:.1f} -- not adapting"
    )


def test_contrast_step_produces_a_transient(encoder):
    encoder.reset()
    for _ in range(200):
        encoder.encode(frame(0.2))
    adapted = np.abs(encoder.encode(frame(0.2))).sum()
    stepped = np.abs(encoder.encode(frame(0.8))).sum()
    assert stepped > adapted * 2, "a contrast step must produce a transient"


def test_stale_frame_still_encodes(encoder):
    encoder.reset()
    stale = Frame(image=np.full((H, W), 0.5, np.float32), timestamp=0.0, stale=True)
    out = encoder.encode(stale)
    assert np.isfinite(out).all()
