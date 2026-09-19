import json

import numpy as np
import pytest

from flybrain.ingest.graph import Graph, GraphHashMismatch, load_graph
from flybrain.paths import ARTIFACTS

GRAPH = ARTIFACTS / "graph_t2.npz"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def graph() -> Graph:
    return load_graph(GRAPH)


def test_neuron_count_exact(graph):
    assert graph.n_neurons == 166_700


def test_csr_is_structurally_valid(graph):
    assert graph.indptr[0] == 0
    assert graph.indptr[-1] == graph.n_edges
    assert np.all(np.diff(graph.indptr) >= 0), "indptr must be non-decreasing"
    assert graph.indices.max() < graph.n_neurons


def test_type_index_returns_expected_counts(graph):
    assert len(graph.type_index("DNp01")) == 2
    assert len(graph.type_index("LPLC2")) == 185
    assert len(graph.type_index("LC4")) == 126
    assert len(graph.type_index("L1")) == 1776


def test_superclass_index_returns_expected_counts(graph):
    assert len(graph.superclass_index("descending_neuron")) == 1314
    assert len(graph.superclass_index("vnc_motor")) == 708


def test_unknown_type_returns_empty(graph):
    assert len(graph.type_index("not_a_real_cell_type")) == 0


def test_hash_mismatch_refuses_to_load(tmp_path, graph):
    """A corrupted artifact must refuse to load -- every result has to be
    traceable to an exact build."""
    data = dict(np.load(GRAPH, allow_pickle=False))
    data["weights"] = data["weights"].copy()
    data["weights"][0] += 1.0
    bad = tmp_path / "graph_t2.npz"
    np.savez(bad, **data)
    manifest = json.loads(GRAPH.with_suffix(".manifest.json").read_text())
    bad.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(GraphHashMismatch):
        load_graph(bad)


def test_missing_manifest_refuses_to_load(tmp_path):
    """When verify=True (default), a missing manifest must raise GraphHashMismatch."""
    data = dict(np.load(GRAPH, allow_pickle=False))
    bad = tmp_path / "graph_t2.npz"
    np.savez(bad, **data)
    # Do NOT write a manifest file
    with pytest.raises(GraphHashMismatch):
        load_graph(bad)


def test_manifest_without_content_hash_refuses_to_load(tmp_path):
    """When verify=True (default), a manifest lacking content_hash key must raise."""
    data = dict(np.load(GRAPH, allow_pickle=False))
    bad = tmp_path / "graph_t2.npz"
    np.savez(bad, **data)
    # Write a manifest without content_hash
    manifest = json.loads(GRAPH.with_suffix(".manifest.json").read_text())
    del manifest["content_hash"]
    bad.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(GraphHashMismatch):
        load_graph(bad)


def test_verify_false_skips_manifest_requirement(tmp_path):
    """When verify=False, loading without a manifest must succeed."""
    data = dict(np.load(GRAPH, allow_pickle=False))
    no_check = tmp_path / "graph_t2.npz"
    np.savez(no_check, **data)
    # Do NOT write a manifest file
    g = load_graph(no_check, verify=False)
    assert g.n_neurons == 166_700
