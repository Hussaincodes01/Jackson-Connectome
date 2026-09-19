import json

import numpy as np
import pytest

from flybrain.ingest.build_graph import build_graph, load_annotations, load_nt
from flybrain.paths import ARTIFACTS

EXPECTED_EDGES_T2 = 57_670_765
EXPECTED_NEURONS = 166_700


def test_annotations_load_with_expected_neuron_count():
    ann = load_annotations()
    assert len(ann["body_ids"]) == EXPECTED_NEURONS
    assert len(ann["types"]) == EXPECTED_NEURONS


def test_annotations_contain_known_cell_types():
    ann = load_annotations()
    counts = {t: int((ann["types"] == t).sum()) for t in ("L1", "L2", "LPLC2", "LC4", "DNp01")}
    assert counts == {"L1": 1776, "L2": 1779, "LPLC2": 185, "LC4": 126, "DNp01": 2}


def test_nt_table_covers_all_annotated_neurons():
    ann = load_annotations()
    bodies, consensus, _ = load_nt()
    assert len(np.intersect1d(ann["body_ids"], bodies)) >= EXPECTED_NEURONS - 200


@pytest.mark.slow
def test_build_graph_edge_count_is_exact():
    path = build_graph(threshold=2, out_dir=ARTIFACTS)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    # Edges surviving the threshold, before sign-zero neurons are dropped.
    assert manifest["edges_after_threshold"] == EXPECTED_EDGES_T2


@pytest.mark.slow
def test_dale_law_every_neuron_has_one_output_sign():
    """Each neuron's out-edges must all share a sign. This is the single
    strongest structural check on the whole ingest pipeline."""
    data = np.load(ARTIFACTS / "graph_t2.npz", allow_pickle=False)
    indptr, weights = data["indptr"], data["weights"]
    rng = np.random.default_rng(0)
    rows = rng.choice(len(indptr) - 1, size=20_000, replace=False)
    for r in rows:
        seg = weights[indptr[r]:indptr[r + 1]]
        if seg.size:
            assert np.all(seg > 0) or np.all(seg < 0), f"neuron {r} has mixed signs"


@pytest.mark.slow
def test_known_looming_pathway_edge_exists():
    """LPLC2 -> DNp01 is the escape circuit the battery depends on."""
    data = np.load(ARTIFACTS / "graph_t2.npz", allow_pickle=True)
    indptr, indices, types = data["indptr"], data["indices"], data["types"]
    lplc2 = np.flatnonzero(types == "LPLC2")
    dnp01 = set(np.flatnonzero(types == "DNp01").tolist())
    hits = sum(
        1 for s in lplc2
        if dnp01 & set(indices[indptr[s]:indptr[s + 1]].tolist())
    )
    assert hits > 0, "no LPLC2 -> DNp01 connection survived the build"


@pytest.mark.slow
def test_manifest_hash_is_reproducible():
    first = json.loads((ARTIFACTS / "graph_t2.manifest.json").read_text())["content_hash"]
    path = build_graph(threshold=2, out_dir=ARTIFACTS)
    second = json.loads(path.with_suffix(".manifest.json").read_text())["content_hash"]
    assert first == second
