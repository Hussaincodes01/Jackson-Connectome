import pytest
import torch
import pyarrow.feather as feather
from flybrain.paths import WEIGHTS, ANNOTATIONS, NEUROTRANSMITTERS, count_edges

EXPECTED_EDGES = 151_856_684
EXPECTED_NEURONS = 166_700


def test_data_files_present():
    for p in (WEIGHTS, ANNOTATIONS, NEUROTRANSMITTERS):
        assert p.exists(), f"missing data file: {p}"


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA required; event-driven core targets GPU"


@pytest.mark.slow
def test_edge_count_exact():
    assert count_edges() == EXPECTED_EDGES


def test_annotated_neuron_count_exact():
    tbl = feather.read_table(ANNOTATIONS, columns=["superclass"])
    n = sum(1 for s in tbl.column("superclass").to_pylist() if s is not None)
    assert n == EXPECTED_NEURONS
