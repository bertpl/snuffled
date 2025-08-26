import math

import numpy as np
import pytest

from snuffled._core.analysis.roots.single_root.helpers_curve_fitting import compute_x_deltas, fit_curve_brute_force
from snuffled._core.utils.noise import noise_from_float


# =================================================================================================
#  Sampling
# =================================================================================================
@pytest.mark.parametrize(
    "dx, k",
    [
        (1e-3, 1),
        (1e-3, 2),
        (1e-6, 3),
        (1e-9, 5),
        (1e-9, 1000),
    ],
)
def test_compute_x_deltas(dx: float, k: int):
    # --- arrange -----------------------------------------
    len_expected = 3 + 6 * k
    seed1 = 1234
    seed2 = 2345

    # --- act ---------------------------------------------
    x_deltas = compute_x_deltas(dx, k, seed=seed1)
    x_deltas_2 = compute_x_deltas(dx, k, seed=seed1)
    x_deltas_3 = compute_x_deltas(dx, k, seed=seed2)

    # --- assert ------------------------------------------

    # check seed handling
    assert np.array_equal(x_deltas, x_deltas_2)
    assert not np.array_equal(x_deltas, x_deltas_3)

    # check counts and sorting
    assert len(x_deltas) == len_expected
    assert np.array_equal(np.sort(x_deltas), x_deltas)

    # check statistical properties
    assert dx / math.sqrt(2) <= min(x_deltas) < dx
    assert 4 * dx < max(x_deltas) <= 4 * dx * math.sqrt(2)

    assert np.median(x_deltas[: len_expected // 3]) == dx
    assert np.median(x_deltas) == 2 * dx
    assert np.median(x_deltas[-(len_expected // 3) :]) == 4 * dx


# =================================================================================================
#  fit_curve_brute_force
# =================================================================================================
@pytest.mark.parametrize("c_noise", [1e-9, 1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("true_c", [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0])
def test_fit_curve_brute_force_c(true_c: float, c_noise: float):
    """Check if for simple cases we can reliably reverse-engineer parameter 'c'"""

    # --- arrange -----------------------------------------
    x_values = np.linspace(0.5, 1.5, 21)
    fx_values = np.array([(x**true_c) + (c_noise * noise_from_float(x)) for x in x_values])

    # --- act ---------------------------------------------
    a_values, b_values, c_values, cost_values = fit_curve_brute_force(
        x=x_values,
        fx=fx_values,
        range_b=(-0.5, 1.0),
        range_c=(0.1, 10),
        c_sign=1.0,
        n_grid=201,
        reg=1e-3,
        tol_c0=0.0,
        tol_c1=2.0,
    )

    print(f"b : {min(b_values)} -> {max(b_values)}")
    print(f"c : {min(c_values)} -> {max(c_values)}")

    # --- assert ------------------------------------------
    # NOTE: we need to add some tolerances, since we search over a fixed grid
    #       that might not even contain the true value.
    assert min(b_values) - 0.01 <= 0.0 <= max(b_values) + 0.01
    assert 0.99 * min(c_values) <= true_c <= 1.01 * max(c_values)


@pytest.mark.parametrize("c_noise", [1e-9, 1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("true_b", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_fit_curve_brute_force_b(true_b: float, c_noise: float):
    """Check if for simple cases we can reliably reverse-engineer parameter 'b'"""

    # --- arrange -----------------------------------------
    x_values = np.linspace(0.5, 1.5, 21)
    fx_values = np.array([true_b + (1 - true_b) * x + (c_noise * noise_from_float(x)) for x in x_values])

    # --- act ---------------------------------------------
    a_values, b_values, c_values, cost_values = fit_curve_brute_force(
        x=x_values,
        fx=fx_values,
        range_b=(-0.5, 1.0),
        range_c=(0.1, 10),
        c_sign=1.0,
        n_grid=201,
        reg=1e-3,
        tol_c0=0.0,
        tol_c1=2.0,
    )

    print(f"b : {min(b_values)} -> {max(b_values)}")
    print(f"c : {min(c_values)} -> {max(c_values)}")

    # --- assert ------------------------------------------
    # NOTE: we need to add some tolerances, since we search over a fixed grid
    #       that might not even contain the true value.
    assert min(b_values) - 0.01 <= true_b <= max(b_values) + 0.01
    assert 0.99 * min(c_values) <= 1.0 <= 1.01 * max(c_values)
