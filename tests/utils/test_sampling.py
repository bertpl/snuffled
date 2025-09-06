import math

import numpy as np
import pytest

from snuffled._core.utils.sampling import (
    fit_fixed_sum_exponential_intervals,
    get_fixed_sum_exponential_intervals,
    multi_scale_samples,
    pseudo_uniform_samples,
    sample_integers,
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


# =================================================================================================
#  Integer sampling
# =================================================================================================
@pytest.mark.parametrize(
    "i_min, i_max, n",
    [
        (0, 100, 10),
        (0, 100, 100),
        (50, 100, 10),
        (50, 100, 50),
        (990, 1_000, 5),
        (999_900, 1_000_000, 10),
        (1_000, 1_000_000, 1_000),
    ],
)
def test_sample_n_integers(i_min: int, i_max: int, n: int):
    # --- act ---------------------------------------------
    samples = sample_integers(i_min, i_max, n)

    # --- assert ------------------------------------------
    assert isinstance(samples, list)
    assert len(samples) == n
    assert all([i_min <= s < i_max for s in samples])
    assert sorted(samples) == samples
    assert len(set(samples)) == n


# =================================================================================================
#  Pseudo-uniform sampling
# =================================================================================================
@pytest.mark.parametrize(
    "x_min, x_max, n",
    [
        (1.0, 7.0, 5),
        (-3.45, 6.78, 101),
        (-101.0, 101.0, 32),
    ],
)
def test_pseudo_uniform_samples(x_min: float, x_max: float, n: int):
    # --- arrange -----------------------------------------
    w = (x_max - x_min) / n  # sub-interval width

    # --- act ---------------------------------------------
    samples = pseudo_uniform_samples(x_min, x_max, n)

    # --- assert ------------------------------------------
    assert len(samples) == n
    assert min(samples) >= x_min
    assert max(samples) < x_max
    assert len({math.floor((x_min - x) / w) for x in samples}) == n


def test_pseudo_uniform_samples_edge_cases():
    # --- act & assert 1 ----------------------------------
    with pytest.raises(ValueError):
        _ = pseudo_uniform_samples(0.0, 1.0, -3)

    # --- act & assert 2 ----------------------------------
    result = pseudo_uniform_samples(0.0, 1.0, 0)
    assert isinstance(result, np.ndarray)
    assert len(result) == 0
