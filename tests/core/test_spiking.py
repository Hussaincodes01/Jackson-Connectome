import numpy as np
import torch

from flybrain.core.lif import LIFNetwork, LIFParams


def two_neuron_net(weight: float, **kw):
    """Neuron 0 -> neuron 1 with the given signed weight."""
    indptr = np.array([0, 1, 1], dtype=np.int64)
    indices = np.array([1], dtype=np.int32)
    weights = np.array([weight], dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams(**kw))


def test_spike_fires_exactly_at_threshold():
    # tau_m is enormous so the membrane update is a no-op; otherwise V decays
    # to -45.34 before the threshold check and this would silently pass for
    # the wrong reason.
    net = two_neuron_net(1.0, tau_m=1e9)
    net.v[0] = -45.0  # exactly v_th
    spikes = net.step()
    assert bool(spikes[0]) is True


def test_no_spike_just_below_threshold():
    net = two_neuron_net(1.0, tau_m=1e9)  # effectively no decay
    net.v[0] = -45.001
    spikes = net.step()
    assert bool(spikes[0]) is False


def test_voltage_resets_after_spike():
    net = two_neuron_net(1.0)
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.v[0]) - (-52.0)) < 1e-6


def test_refractory_lasts_exactly_ceil_tref_over_dt_steps():
    """t_ref=2.2, dt=1.0 -> n_ref = 2 steps clamped at reset."""
    net = two_neuron_net(1.0, t_ref=2.2, dt=1.0)
    assert net.n_ref == 2
    net.v[0] = -40.0
    net.step()                       # spikes, refrac = 2
    for expected in (1, 0):
        net.step(torch.tensor([100.0, 0.0]))
        assert abs(float(net.v[0]) - (-52.0)) < 1e-6, "clamped during refractory"
        assert int(net.refrac[0]) == expected
    # Free now: the same drive reaches threshold and fires. (Checking the
    # spike, not the voltage -- a spike resets V to v_rest, so asserting
    # v > v_rest here would assert the opposite of correct behaviour.)
    spikes = net.step(torch.tensor([100.0, 0.0]))
    assert bool(spikes[0]) is True


def test_excitatory_edge_raises_postsynaptic_current():
    net = two_neuron_net(+3.0)
    net.v[0] = -40.0
    net.step()
    assert float(net.i_syn[1]) > 0
    assert abs(float(net.i_syn[1]) - 3.0) < 1e-6


def test_inhibitory_edge_lowers_postsynaptic_current():
    net = two_neuron_net(-3.0)
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.i_syn[1]) - (-3.0)) < 1e-6


def test_postsynaptic_response_arrives_one_step_later():
    """Propagation happens after the membrane update, so a synapse costs one
    timestep. This is what the Phase 4 latency assertion measures."""
    net = two_neuron_net(+5.0)
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.v[1]) - (-52.0)) < 1e-6, "target unchanged on the spike step"
    net.step()
    assert float(net.v[1]) > -52.0, "target responds on the following step"


def test_spike_fraction_is_reported():
    net = two_neuron_net(1.0)
    net.v[0] = -40.0
    net.step()
    assert abs(net.last_spike_fraction - 0.5) < 1e-9


def test_fan_out_sums_into_multiple_targets():
    indptr = np.array([0, 2, 2, 2], dtype=np.int64)
    indices = np.array([1, 2], dtype=np.int32)
    weights = np.array([2.0, -4.0], dtype=np.float32)
    net = LIFNetwork(indptr, indices, weights, LIFParams())
    net.v[0] = -40.0
    net.step()
    assert abs(float(net.i_syn[1]) - 2.0) < 1e-6
    assert abs(float(net.i_syn[2]) - (-4.0)) < 1e-6
