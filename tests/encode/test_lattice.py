import numpy as np
import pytest

from flybrain.encode.lattice import build_lattice, sampling_matrix
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph():
    return load_graph(ARTIFACTS / "graph_t2.npz")


def test_artifact_carries_soma_side(graph):
    assert hasattr(graph, "soma_side")
    l1 = graph.type_index("L1")
    sides = graph.soma_side[l1]
    assert int((sides == "R").sum()) == 892
    assert int((sides == "L").sum()) == 884


def test_right_l1_lattice_has_892_columns(graph):
    lat = build_lattice(graph, "L1", "R")
    assert len(lat.neuron_idx) == 892


def test_lattice_excludes_neurons_without_coordinates(graph):
    lat = build_lattice(graph, "L1", "R")
    assert not np.isnan(lat.hex1).any()
    assert not np.isnan(lat.hex2).any()


def test_lattice_xy_is_normalised(graph):
    lat = build_lattice(graph, "L1", "R")
    assert lat.xy.shape == (len(lat.neuron_idx), 2)
    assert lat.xy.min() >= 0.0 and lat.xy.max() <= 1.0
    # A hex lattice must actually span both axes, not collapse to a line.
    assert np.ptp(lat.xy[:, 0]) > 0.9 and np.ptp(lat.xy[:, 1]) > 0.9


def test_neighbouring_columns_are_near_in_xy(graph):
    """Retinotopy must be preserved: hex neighbours map to nearby points."""
    lat = build_lattice(graph, "L1", "R")
    h1, h2, xy = lat.hex1, lat.hex2, lat.xy
    pairs = 0
    for i in range(0, len(h1), 37):
        neighbour = np.flatnonzero((h1 == h1[i] + 1) & (h2 == h2[i]))
        if neighbour.size:
            j = neighbour[0]
            far = np.flatnonzero((h1 == h1[i]) & (h2 == h2[i] + 10))
            if far.size:
                assert np.linalg.norm(xy[i] - xy[j]) < np.linalg.norm(xy[i] - xy[far[0]])
                pairs += 1
    assert pairs > 5, "not enough neighbour pairs tested"


def test_sampling_matrix_rows_are_normalised(graph):
    lat = build_lattice(graph, "L1", "R")
    m = sampling_matrix(lat, 120, 160)
    assert m.shape == (892, 120 * 160)
    sums = np.asarray(m.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_uniform_image_yields_uniform_column_response(graph):
    lat = build_lattice(graph, "L1", "R")
    m = sampling_matrix(lat, 120, 160)
    frame = np.full(120 * 160, 0.5, dtype=np.float32)
    out = m @ frame
    assert np.allclose(out, 0.5, atol=1e-4)


def test_left_half_bright_image_activates_one_side_of_the_lattice(graph):
    """A half-bright frame must produce a spatial gradient across columns."""
    lat = build_lattice(graph, "L1", "R")
    m = sampling_matrix(lat, 120, 160)
    img = np.zeros((120, 160), dtype=np.float32)
    img[:, :80] = 1.0
    out = m @ img.ravel()
    left_cols = out[lat.xy[:, 0] < 0.35]
    right_cols = out[lat.xy[:, 0] > 0.65]
    assert left_cols.mean() > right_cols.mean() + 0.3
