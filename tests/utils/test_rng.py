from collections import Counter

import numpy as np
import pytest

from snuffled._core.utils.rng import _xoroshiro128plus_next, new_rng_state, rand_float64, rand_int64, shuffle_in_place
from tests.helpers import only_with_numba_jit


# =================================================================================================
#  Pinned values: CI runs these tests with the JIT on and off, and both must give the same values
# =================================================================================================
@pytest.mark.parametrize(
    "seed, expected_rng_state",
    [
        (42, [13679457532755275413, 2949826092126892291]),
        (-1, [16490336266968443936, 16834447057089888969]),
    ],
)
def test_new_rng_state_pinned(seed: int, expected_rng_state: list[int]):
    """A seed, negative or not, gives its pinned uint64 state."""
    # --- act --------------------------
    rng_state = new_rng_state(seed)

    # --- assert -----------------------
    assert rng_state.dtype == np.uint64
    assert [int(v) for v in rng_state] == expected_rng_state


def test_rand_float64_pinned():
    """The first 3 float64 values drawn from seed 42 match the pinned values."""
    # --- arrange ----------------------
    rng_state = new_rng_state(42)

    # --- act --------------------------
    values = [rand_float64(rng_state) for _ in range(3)]

    # --- assert -----------------------
    assert values == [0.9014752716487433, 0.07700507558070205, 0.5295414206238577]


def test_shuffle_in_place_pinned():
    """Shuffling 0..9 with seed 42 gives the pinned order."""
    # --- arrange ----------------------
    rng_state = new_rng_state(42)
    values = np.arange(10, dtype=np.float64)

    # --- act --------------------------
    shuffle_in_place(rng_state, values)

    # --- assert -----------------------
    assert values.tolist() == [4, 3, 8, 5, 2, 0, 9, 6, 7, 1]


# =================================================================================================
#  Declared numba signatures: without the JIT they do not apply, so these run with the JIT only
# =================================================================================================
@only_with_numba_jit
def test_shuffle_in_place_rejects_other_dtypes():
    """An int64 array does not match the declared float64 signature, so the call raises."""
    # --- arrange ----------------------
    values = np.arange(3, dtype=np.int64)

    # --- act / assert -----------------
    with pytest.raises(TypeError, match="No matching definition"):
        shuffle_in_place(new_rng_state(1), values)


# =================================================================================================
#  new_rng_state
# =================================================================================================
def test_new_rng_state_depends_on_seed():
    """The same seed gives the same state, and a different seed a different one."""
    # --- act / assert -----------------
    assert np.array_equal(new_rng_state(1), new_rng_state(1))
    assert not np.array_equal(new_rng_state(1), new_rng_state(3))


# =================================================================================================
#  rand_float64
# =================================================================================================
def test_rand_float64_stats():
    """The drawn values spread uniformly over [0.0, 1.0), without repeats."""
    # --- arrange ----------------------
    rng_state = new_rng_state(1)

    # --- act --------------------------
    values = [rand_float64(rng_state) for _ in range(10_000)]

    # --- assert -----------------------
    assert min(values) < 0.01
    assert max(values) > 0.99
    assert 0.49 < np.mean(values) < 0.51
    assert len(set(values)) == len(values)  # 10,000 draws out of 2^53 possible values almost never repeat


@pytest.mark.parametrize(
    "rng_state, expected_uint64, expected_float64",
    [
        ([0, 0], 0, 0.0),
        ([np.iinfo(np.uint64).max, 0], np.iinfo(np.uint64).max, 1.0 - np.finfo(np.float64).eps),
    ],
    ids=["min_value", "max_value"],
)
def test_rand_float64_range_endpoints(rng_state: list[int], expected_uint64: int, expected_float64: float):
    """The smallest and largest uint64 values map to 0.0 and to 1.0 - eps."""
    # --- arrange ----------------------
    rng_state_np = np.array(rng_state, dtype=np.uint64)

    # --- act --------------------------
    value_uint64 = _xoroshiro128plus_next(rng_state_np.copy())
    value_float64 = rand_float64(rng_state_np)

    # --- assert -----------------------
    assert value_uint64 == expected_uint64
    assert value_float64 == expected_float64


# =================================================================================================
#  rand_int64
# =================================================================================================
@pytest.mark.parametrize("low, high", [(0, 100), (-50, 50)])
def test_rand_int64_covers_range(low: int, high: int):
    """The drawn values cover every integer in [low, high) and no other, with the expected mean."""
    # --- arrange ----------------------
    rng_state = new_rng_state(1)

    # --- act --------------------------
    values = [rand_int64(rng_state, low, high) for _ in range(10_000)]

    # --- assert -----------------------
    assert set(values) == set(range(low, high))
    assert abs(np.mean(values) - (low + high - 1) / 2) < 1.0


def test_rand_int64_large_range():
    """Values drawn from a range of 10^9 integers stay in range, spread out, and rarely repeat."""
    # --- arrange ----------------------
    rng_state = new_rng_state(1)

    # --- act --------------------------
    values = [rand_int64(rng_state, 0, 1_000_000_000) for _ in range(10_000)]

    # --- assert -----------------------
    assert all(0 <= v < 1_000_000_000 for v in values)
    assert min(values) < 100_000_000
    assert max(values) > 900_000_000
    assert len(set(values)) >= 9_999


# =================================================================================================
#  shuffle_in_place
# =================================================================================================
@pytest.mark.parametrize("n", [0, 1, 2, 10])
def test_shuffle_in_place_permutes(n: int):
    """A shuffle keeps the same values, in an order that depends only on the seed."""
    # --- arrange ----------------------
    values_1 = np.arange(n, dtype=np.float64)
    values_2 = np.arange(n, dtype=np.float64)

    # --- act --------------------------
    shuffle_in_place(new_rng_state(7), values_1)
    shuffle_in_place(new_rng_state(7), values_2)

    # --- assert -----------------------
    assert sorted(values_1.tolist()) == list(range(n))
    assert np.array_equal(values_1, values_2)


def test_shuffle_in_place_is_uniform():
    """Each of the 6 orders of 3 values comes out about equally often."""
    # --- arrange ----------------------
    rng_state = new_rng_state(1)

    # --- act --------------------------
    counts: Counter[tuple[float, ...]] = Counter()
    for _ in range(6_000):
        values = np.arange(3, dtype=np.float64)
        shuffle_in_place(rng_state, values)
        counts[tuple(values.tolist())] += 1

    # --- assert -----------------------
    assert len(counts) == 6
    assert all(800 < count < 1200 for count in counts.values())
