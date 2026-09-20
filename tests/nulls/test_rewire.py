import numpy as np
import pytest

from flybrain.nulls.rewire import (
    NULL_MODELS,
    degree_preserving_rewire,
    sign_permutation,
    type_preserving_rewire,
    weight_permutation,
)


@pytest.fixture
def toy():
    """6 neurons. Rows 0-2 excitatory, rows 3-5 inhibitory (Dale's law holds)."""
    indptr = np.array([0, 2, 4, 5, 7, 8, 9], dtype=np.int64)
    indices = np.array([1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=np.int32)
    weights = np.array([2.0, 3.0, 1.0, 5.0, 4.0, -2.0, -6.0, -1.0, -3.0], dtype=np.float32)
    types = np.array(["A", "A", "A", "B", "B", "B"], dtype="<U32")
    return indptr, indices, weights, types


def out_degrees(indptr):
    return np.diff(indptr)


def in_degrees(indices, n):
    return np.bincount(indices, minlength=n)


def dale_holds(indptr, weights):
    for r in range(len(indptr) - 1):
        seg = weights[indptr[r]:indptr[r + 1]]
        if seg.size and not (np.all(seg > 0) or np.all(seg < 0)):
            return False
    return True


def test_all_four_models_are_registered():
    assert set(NULL_MODELS) == {
        "weight_permutation",
        "degree_preserving",
        "sign_permutation",
        "type_preserving",
    }


def test_weight_permutation_preserves_topology_and_magnitude_multiset(toy):
    indptr, indices, weights, _ = toy
    p, i, w = weight_permutation(indptr, indices, weights, seed=0)
    assert np.array_equal(p, indptr)
    assert np.array_equal(i, indices)
    assert np.allclose(np.sort(np.abs(w)), np.sort(np.abs(weights)))
    assert dale_holds(p, w)


def test_weight_permutation_actually_changes_something(toy):
    indptr, indices, weights, _ = toy
    _, _, w = weight_permutation(indptr, indices, weights, seed=0)
    assert not np.allclose(w, weights)


def test_degree_preserving_keeps_both_degree_sequences(toy):
    indptr, indices, weights, _ = toy
    n = len(indptr) - 1
    p, i, w = degree_preserving_rewire(indptr, indices, weights, seed=0)
    assert np.array_equal(out_degrees(p), out_degrees(indptr))
    assert np.array_equal(
        np.sort(in_degrees(i, n)), np.sort(in_degrees(indices, n))
    )
    assert dale_holds(p, w)


def test_degree_preserving_rewires_targets(toy):
    indptr, indices, weights, _ = toy
    _, i, _ = degree_preserving_rewire(indptr, indices, weights, seed=3)
    assert not np.array_equal(i, indices)


def test_sign_permutation_preserves_topology_and_sign_counts(toy):
    indptr, indices, weights, _ = toy
    p, i, w = sign_permutation(indptr, indices, weights, seed=0)
    assert np.array_equal(i, indices)
    assert dale_holds(p, w)
    # The number of excitatory and inhibitory neurons is unchanged.
    def neuron_signs(ptr, ws):
        return sorted(
            np.sign(ws[ptr[r]]) for r in range(len(ptr) - 1) if ptr[r + 1] > ptr[r]
        )
    assert neuron_signs(p, w) == neuron_signs(indptr, weights)


def test_type_preserving_keeps_type_to_type_counts(toy):
    indptr, indices, weights, types = toy
    p, i, w = type_preserving_rewire(indptr, indices, weights, types, seed=0)

    def block_counts(ptr, idx):
        counts: dict[tuple[str, str], int] = {}
        for r in range(len(ptr) - 1):
            for t in idx[ptr[r]:ptr[r + 1]]:
                key = (types[r], types[t])
                counts[key] = counts.get(key, 0) + 1
        return counts

    assert block_counts(p, i) == block_counts(indptr, indices)
    assert dale_holds(p, w)


def test_all_models_are_reproducible_for_a_seed(toy):
    indptr, indices, weights, types = toy
    for name, fn in NULL_MODELS.items():
        args = (indptr, indices, weights, types, 11) if name == "type_preserving" else (
            indptr, indices, weights, 11
        )
        a = fn(*args)
        b = fn(*args)
        for x, y in zip(a, b):
            assert np.array_equal(x, y), f"{name} is not reproducible"


def test_models_do_not_mutate_their_input(toy):
    indptr, indices, weights, types = toy
    before = weights.copy()
    weight_permutation(indptr, indices, weights, seed=0)
    degree_preserving_rewire(indptr, indices, weights, seed=0)
    sign_permutation(indptr, indices, weights, seed=0)
    type_preserving_rewire(indptr, indices, weights, types, seed=0)
    assert np.array_equal(weights, before)


def test_edge_count_is_conserved_by_every_model(toy):
    indptr, indices, weights, types = toy
    for name, fn in NULL_MODELS.items():
        args = (indptr, indices, weights, types, 5) if name == "type_preserving" else (
            indptr, indices, weights, 5
        )
        _, i, w = fn(*args)
        assert len(i) == len(indices), name
        assert len(w) == len(weights), name
