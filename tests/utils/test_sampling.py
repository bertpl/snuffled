import numpy as np
import pytest

from snuffled._core.utils.sampling import (
    fit_fixed_sum_exponential_intervals,
    get_fixed_sum_exponential_intervals,
    get_fixed_sum_integers,
    multi_scale_samples,
)


# =================================================================================================
#  Multi-scale sampling
# =================================================================================================
@pytest.mark.parametrize(
    "x_min, x_max, dx_min, n",
    [
        (-1.0, 2.0, 1e-3, 100),
        (-1.0, 2.0, 1e-6, 100),
        (-1.0, 2.0, 1e-9, 10),
        (-1.0, 2.0, 1e-9, 100),
        (-1.0, 2.0, 1e-9, 1000),
        (-1.0, 2.0, 1e-9, 10000),
    ],
)
def test_multi_scale_sampling(x_min: float, x_max: float, dx_min: float, n: int):
    # --- act ---------------------------------------------
    samples = multi_scale_samples(x_min, x_max, dx_min, n)

    # --- assert ------------------------------------------
    assert samples[0] == x_min
    assert samples[-1] == x_max
    assert len(samples) == n
    assert len(set(samples)) == n
    assert list(samples) == sorted(samples)
    assert all(x_min <= s <= x_max for s in samples)
    assert len(set(np.diff(samples))) == n - 1
    assert min(np.diff(samples)) == pytest.approx(dx_min, rel=1e-15)


def test_multi_scale_sampling_seed():
    # --- act ---------------------------------------------
    s_1 = multi_scale_samples(-10.0, 10.0, 1e-6, 1000, seed=123)
    s_2 = multi_scale_samples(-10.0, 10.0, 1e-6, 1000, seed=123)  # same seed
    s_3 = multi_scale_samples(-10.0, 10.0, 1e-6, 1000, seed=345)  # different seed

    # --- assert ------------------------------------------
    assert np.array_equal(s_1, s_2)
    assert not np.array_equal(s_1, s_3)


# =================================================================================================
#  Constrained integer sampling
# =================================================================================================
@pytest.mark.parametrize("n", [5, 10, 100])
@pytest.mark.parametrize(
    "v_min, v_max",
    [
        (5, 5),
        (4, 6),
        (2, 8),
        (10, 10),
        (9, 11),
        (5, 15),
        (100, 100),
        (99, 101),
        (90, 110),
        (50, 150),
    ],
)
def test_fixed_sum_integers(n: int, v_min: int, v_max: int):
    # --- arrange -----------------------------------------
    tgt_sum = round(0.5 * (v_min + v_max) * n)

    # --- act ---------------------------------------------
    integers = get_fixed_sum_integers(n, v_min, v_max, tgt_sum)

    # --- assert ------------------------------------------
    assert all(v_min <= v <= v_max for v in integers)
    assert sum(integers) == tgt_sum


def test_fixed_sum_integers_seed():
    # --- arrange -----------------------------------------
    n = 100
    v_min = 5
    v_max = 15
    tgt_sum = 1000

    # --- act ---------------------------------------------
    v_1 = get_fixed_sum_integers(n, v_min, v_max, tgt_sum, seed=10)
    v_2 = get_fixed_sum_integers(n, v_min, v_max, tgt_sum, seed=10)  # same seed
    v_3 = get_fixed_sum_integers(n, v_min, v_max, tgt_sum, seed=11)  # different seed

    # --- assert ------------------------------------------
    assert np.array_equal(v_1, v_2)
    assert not np.array_equal(v_1, v_3)


# =================================================================================================
#  Exponential sampling
# =================================================================================================
@pytest.mark.parametrize(
    "n, tgt_sum, dx_min",
    [
        (-1, 100.0, 1.0),
        (0, 100.0, 1.0),
        (1, 100.0, 1.0),
        (2, -1.0, 1.0),
        (2, 0.0, 1.0),
        (2, 100.0, -1.0),
        (2, 100.0, 0.0),
        (100, 100.0, 10.0),
    ],
)
def test_fit_fixed_sum_exponential_intervals_bad_args(n: int, tgt_sum: float, dx_min: float):
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = fit_fixed_sum_exponential_intervals(n, tgt_sum, dx_min)


@pytest.mark.parametrize(
    "n, tgt_sum, dx_min, expected_c",
    [
        (100, 100.0, 1.0, 1.0),
        (2, 100.0, 1.0, 99.0),
        (2, 10.0, 1.0, 9.0),
        (2, 3.0, 1.0, 2.0),
        (3, 7.0, 1.0, 2.0),
        (4, 15.0, 1.0, 2.0),
        (3, 4.75, 1.0, 1.5),
    ],
)
def test_fit_fixed_sum_exponential_intervals_simple_cases(n: int, tgt_sum: float, dx_min: float, expected_c: float):
    # --- act ---------------------------------------------
    c = fit_fixed_sum_exponential_intervals(n, tgt_sum, dx_min)

    # --- assert ------------------------------------------
    assert c == pytest.approx(expected_c, rel=1e-15)


@pytest.mark.parametrize(
    "n, tgt_sum, dx_min",
    [
        (100, 100.0, 1e-2),
        (200, 100.0, 1e-4),
        (500, 100.0, 1e-6),
        (1_000, 100.0, 1e-8),
        (1_000, 100.0, 1e-10),
        (1_000, 100.0, 1e-12),
        (1_000, 100.0, 1e-14),
        (1_000, 100.0, 1e-16),
        (1_000, 100.0, 1e-24),
        (1_000, 100.0, 1e-32),
        (1_000, 100.0, 1e-64),
        (1_000, 100.0, 1e-96),
        (1_000, 100.0, 1e-128),
        (2_000, 100.0, 1e-128),
        (4_000, 100.0, 1e-128),
        (8_000, 100.0, 1e-128),
        (16_000, 100.0, 1e-128),
        (100_000, 100.0, 1e-4),
    ],
)
def test_fit_and_get_fixed_sum_exponential_intervals_realistic_cases(n: int, tgt_sum: float, dx_min: float):
    # --- act ---------------------------------------------
    c = fit_fixed_sum_exponential_intervals(n, tgt_sum, dx_min)
    interval_sizes = get_fixed_sum_exponential_intervals(n, tgt_sum, dx_min)

    # --- assert ------------------------------------------

    # check result of fit_...
    computed_interval_sizes = [dx_min * (c**i) for i in range(n)]
    assert sum(computed_interval_sizes) == pytest.approx(tgt_sum, rel=n * 1e-15)

    # check result of get_...
    assert sum(interval_sizes) == pytest.approx(tgt_sum, rel=n * 1e-15)
    assert np.allclose(interval_sizes, computed_interval_sizes, rtol=1e-15)
