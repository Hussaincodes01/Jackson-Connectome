import pytest

from flybrain.core.runaway import RunawayDetector, RunawayHalt


def test_quiet_activity_never_trips():
    d = RunawayDetector(ceiling=0.15, window=5)
    for _ in range(100):
        assert d.update(0.02) == "ok"
    assert d.tripped is False


def test_single_spike_above_ceiling_does_not_trip():
    """One busy millisecond is normal; sustained saturation is not."""
    d = RunawayDetector(ceiling=0.15, window=5)
    assert d.update(0.9) == "ok"
    assert d.tripped is False


def test_trips_after_exactly_window_consecutive_breaches():
    d = RunawayDetector(ceiling=0.15, window=5)
    for i in range(4):
        assert d.update(0.20) == "ok", f"tripped early at breach {i + 1}"
    assert d.update(0.20) == "clamp"
    assert d.tripped is True


def test_counter_resets_on_a_quiet_step():
    d = RunawayDetector(ceiling=0.15, window=3)
    d.update(0.2)
    d.update(0.2)
    d.update(0.01)
    assert d.update(0.2) == "ok"


def test_halt_mode_returns_halt():
    d = RunawayDetector(ceiling=0.15, window=2, mode="halt")
    d.update(0.5)
    assert d.update(0.5) == "halt"


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        RunawayDetector(mode="explode")
