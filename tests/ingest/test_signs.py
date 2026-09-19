import numpy as np
import pytest
from flybrain.ingest.signs import load_sign_map, resolve_signs
from flybrain.paths import ROOT


@pytest.fixture
def sign_map():
    return load_sign_map(ROOT / "config" / "nt_signs.yaml")


def test_sign_map_exact_values(sign_map):
    assert sign_map["acetylcholine"] == 1
    assert sign_map["gaba"] == -1
    assert sign_map["glutamate"] == -1
    assert sign_map["histamine"] == -1
    assert sign_map["dopamine"] == 0
    assert sign_map["octopamine"] == 0
    assert sign_map["serotonin"] == 0


def test_each_transmitter_resolves_to_its_sign(sign_map):
    nts = np.array(
        ["acetylcholine", "gaba", "glutamate", "histamine",
         "dopamine", "octopamine", "serotonin"],
        dtype=object,
    )
    empty = np.array([None] * 7, dtype=object)
    assert resolve_signs(nts, empty, sign_map).tolist() == [1, -1, -1, -1, 0, 0, 0]


def test_unclear_falls_back_to_celltype_consensus(sign_map):
    consensus = np.array(["unclear", "unclear"], dtype=object)
    celltype = np.array(["gaba", "acetylcholine"], dtype=object)
    assert resolve_signs(consensus, celltype, sign_map).tolist() == [-1, 1]


def test_missing_everywhere_resolves_to_zero(sign_map):
    consensus = np.array([None, "unclear"], dtype=object)
    celltype = np.array([None, "unclear"], dtype=object)
    assert resolve_signs(consensus, celltype, sign_map).tolist() == [0, 0]


def test_unknown_transmitter_name_resolves_to_zero(sign_map):
    consensus = np.array(["kryptonite"], dtype=object)
    celltype = np.array([None], dtype=object)
    assert resolve_signs(consensus, celltype, sign_map).tolist() == [0]


def test_returns_int8(sign_map):
    out = resolve_signs(
        np.array(["acetylcholine"], dtype=object),
        np.array([None], dtype=object),
        sign_map,
    )
    assert out.dtype == np.int8
