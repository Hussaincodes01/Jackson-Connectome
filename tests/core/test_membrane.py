import math

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from flybrain.core.lif import LIFNetwork, LIFParams


def empty_net(n=3, **kw):
    """A network with no edges -- isolates membrane dynamics."""
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(0, dtype=np.int32)
    weights = np.zeros(0, dtype=np.float32)
    return LIFNetwork(indptr, indices, weights, LIFParams(**kw))


def test_starts_at_resting_potential():
    net = empty_net()
    assert torch.allclose(net.v, torch.full((3,), -52.0, dtype=net.v.dtype))


def test_membrane_decay_matches_closed_form():
    """V(t) = V_rest + (V0 - V_rest) * exp(-t / tau_m), to 1e-6."""
    params = LIFParams(tau_m=20.0, dt=1.0)
    net = empty_net(n=1, tau_m=20.0, dt=1.0, v_th=1e9)
    v0 = -30.0
    net.v[:] = v0
    for n_steps in range(1, 101):
        net.step()
        t = n_steps * params.dt
        expected = params.v_rest + (v0 - params.v_rest) * math.exp(-t / params.tau_m)
        assert abs(float(net.v[0]) - expected) < 1e-6, f"diverged at step {n_steps}"


def test_constant_input_drives_v_to_rest_plus_input():
    """With tau_syn -> 0 the steady state is V_rest + I. Checked at 500 ms."""
    net = empty_net(n=1, tau_syn=1e-6, tau_m=20.0, dt=1.0, v_th=1e9)
    drive = torch.tensor([5.0])
    for _ in range(500):
        net.step(drive)
    assert abs(float(net.v[0]) - (-52.0 + 5.0)) < 1e-3


def test_synaptic_current_decays_with_tau_syn():
    net = empty_net(n=1, tau_syn=5.0, dt=1.0, v_th=1e9)
    net.step(torch.tensor([10.0]))
    first = float(net.i_syn[0])
    net.step()
    second = float(net.i_syn[0])
    assert abs(second / first - math.exp(-1.0 / 5.0)) < 1e-6


def test_reset_restores_initial_state():
    net = empty_net(n=4)
    net.step(torch.full((4,), 3.0))
    net.reset()
    assert torch.allclose(net.v, torch.full((4,), -52.0, dtype=net.v.dtype))
    assert torch.allclose(net.i_syn, torch.zeros(4, dtype=net.i_syn.dtype))


@settings(max_examples=50, deadline=None)
@given(
    tau_m=st.floats(min_value=1.0, max_value=100.0),
    v0=st.floats(min_value=-80.0, max_value=-46.0),
)
def test_decay_is_monotone_toward_rest_for_any_tau(tau_m, v0):
    net = empty_net(n=1, tau_m=tau_m, dt=1.0, v_th=1e9)
    net.v[:] = v0
    previous = v0
    for _ in range(50):
        net.step()
        current = float(net.v[0])
        if v0 > -52.0:
            assert current <= previous + 1e-9
        else:
            assert current >= previous - 1e-9
        previous = current
