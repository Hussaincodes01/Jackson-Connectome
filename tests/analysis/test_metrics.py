import numpy as np
import pytest

from flybrain.analysis.metrics import (
    bootstrap_ci,
    direction_selectivity_index,
    intervals_overlap,
    looming_discrimination,
)


def test_dsi_is_one_for_a_perfectly_selective_response():
    assert direction_selectivity_index(10.0, 0.0) == pytest.approx(1.0)


def test_dsi_is_zero_for_an_unselective_response():
    assert direction_selectivity_index(10.0, 10.0) == pytest.approx(0.0)


def test_dsi_is_negative_when_the_opposite_direction_wins():
    assert direction_selectivity_index(2.0, 8.0) == pytest.approx(-0.6)


def test_dsi_of_silence_is_zero_not_nan():
    assert direction_selectivity_index(0.0, 0.0) == 0.0


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(5.0, 1.0, size=200)
    low, high = bootstrap_ci(values, seed=0)
    assert low < values.mean() < high


def test_bootstrap_ci_is_reproducible():
    values = np.arange(50, dtype=float)
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)


def test_tighter_data_gives_a_narrower_interval():
    rng = np.random.default_rng(1)
    wide = bootstrap_ci(rng.normal(0, 5.0, 200), seed=0)
    tight = bootstrap_ci(rng.normal(0, 0.1, 200), seed=0)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


def test_overlap_detection():
    assert intervals_overlap((0.0, 1.0), (0.5, 2.0)) is True
    assert intervals_overlap((0.0, 1.0), (1.5, 2.0)) is False
    assert intervals_overlap((1.5, 2.0), (0.0, 1.0)) is False


def test_looming_discrimination_is_positive_when_loom_drives_more():
    loom = np.array([1, 2, 8, 20], dtype=float)
    control = np.array([1, 1, 1, 1], dtype=float)
    assert looming_discrimination(loom, control) > 0


def test_looming_discrimination_is_zero_for_identical_responses():
    same = np.array([3, 3, 3], dtype=float)
    assert looming_discrimination(same, same) == pytest.approx(0.0)
