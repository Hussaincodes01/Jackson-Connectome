import numpy as np
import pytest
import torch

from flybrain.core.lif import LIFNetwork, LIFParams
from flybrain.core.subgraph import extract_subgraph
from flybrain.ingest.graph import load_graph
from flybrain.paths import ARTIFACTS

pytestmark = pytest.mark.slow

N_KEEP = 20_000
N_STEPS = 500


@pytest.fixture(scope="module")
def subgraph():
    g = load_graph(ARTIFACTS / "graph_t2.npz")
    rng = np.random.default_rng(7)
    keep = np.sort(rng.choice(g.n_neurons, size=N_KEEP, replace=False))
    return extract_subgraph(g, keep)


def _run(subgraph, device, drive):
    indptr, indices, weights = subgraph
    net = LIFNetwork(indptr, indices, weights, LIFParams(), device=device)
    torch.manual_seed(0)
    raster = np.zeros((N_STEPS, N_KEEP), dtype=bool)
    for t in range(N_STEPS):
        spikes = net.step(drive.to(device))
        raster[t] = spikes.cpu().numpy()
    return raster, net.v.cpu().numpy()


def test_subgraph_is_structurally_valid(subgraph):
    indptr, indices, weights = subgraph
    assert len(indptr) == N_KEEP + 1
    assert indptr[0] == 0 and indptr[-1] == len(indices)
    assert len(indices) == len(weights)
    if len(indices):
        assert indices.max() < N_KEEP


def test_gpu_matches_cpu(subgraph):
    torch.manual_seed(0)
    drive = torch.rand(N_KEEP) * 2.0
    cpu_raster, cpu_v = _run(subgraph, "cpu", drive)
    gpu_raster, gpu_v = _run(subgraph, "cuda", drive)

    agreement = (cpu_raster == gpu_raster).mean()
    assert agreement > 0.995, f"spike rasters agree only {agreement:.4%}"

    total_cpu, total_gpu = cpu_raster.sum(), gpu_raster.sum()
    assert total_cpu > 0, "stimulus produced no spikes -- test is vacuous"
    assert abs(total_cpu - total_gpu) / total_cpu < 0.01

    assert np.max(np.abs(cpu_v - gpu_v)) < 1e-2
