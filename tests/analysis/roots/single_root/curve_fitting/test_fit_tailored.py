import math

import numpy as np
import pytest

from snuffled._core.analysis.roots.single_root.curve_fitting import compute_x_deltas, fitting_cost, fitting_curve
from snuffled._core.analysis.roots.single_root.curve_fitting._curves_and_costs import compute_threshold_cost
from snuffled._core.analysis.roots.single_root.curve_fitting._fit_tailored import (
    fit_curve_exact_three_points,
    fit_curve_tailored,
    fit_curve_with_uncertainty_tailored,
    initial_params,
    param_step,
)
from snuffled._core.utils.noise import noise_from_float


# =================================================================================================
#  TEST - Find solution WITH uncertainty
# =================================================================================================
@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [0.0, 0.2, 0.9])
@pytest.mark.parametrize("c_true", [0.5, 1.0, 2.0, -2.0, -1.0, -0.5])
@pytest.mark.parametrize("c_noise", [0.0, 1e-9, 1e-6, 1e-3, 1.0])
@pytest.mark.parametrize("rel_uncertainty_size", [0.5, 0.75, 1.0])
def test_fit_curve_with_uncertainty_tailored(
    a_true: float, b_true: float, c_true: float, c_noise: float, rel_uncertainty_size: float
):
    """Basic checks if all basic invariants are respected in the returned values."""

    # --- arrange -----------------------------------------

    # define optimization problem
    x_values = compute_x_deltas(dx=0.5, k=5, seed=42)
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)
    for i, x in enumerate(x_values):
        fx_values[i] += c_noise * noise_from_float(x)

    reg = 1e-3
    range_a = (0.1, 10.0)
    range_b = (-0.5, 1.0)
    if c_true > 0.0:
        range_c = (0.25, 4.0)
    else:
        range_c = (-4.0, -0.25)

    # compute reference values to check correctness
    fx_q25, fx_q50, fx_q75 = np.quantile(fx_values, [0.25, 0.50, 0.75])
    a_opt, b_opt, c_opt = fit_curve_tailored(x_values, fx_values, range_a, range_b, range_c, reg)
    cost_opt = fitting_cost(x_values, fx_values, a_opt, b_opt, c_opt, reg)
    threshold_cost = compute_threshold_cost(rel_uncertainty_size, cost_opt, fx_q25, fx_q50, fx_q75)

    # --- act ---------------------------------------------
    a_values, b_values, c_values, cost_values = fit_curve_with_uncertainty_tailored(
        x_values,
        fx_values,
        range_a,
        range_b,
        range_c,
        reg,
        rel_uncertainty_size=rel_uncertainty_size,
        debug_flag=True,
    )

    # --- assert ------------------------------------------

    # check if (a,b,c) vs cost are mutually correct
    for a, b, c, cost in zip(a_values, b_values, c_values, cost_values):
        expected_cost = fitting_cost(x_values, fx_values, a, b, c, reg)
        assert cost == expected_cost

    # check if all (a,b,c) are within range
    for a, b, c in zip(a_values, b_values, c_values):
        assert range_a[0] <= a <= range_a[1]
        assert range_b[0] <= b <= range_b[1]
        assert range_c[0] <= c <= range_c[1]

    # check if cost_values are as expected
    assert (
        min(cost_values) <= cost_opt
    )  # <= not ==, because we can serendipitously a better optimum during uncertainty exploration
    assert 0.9 * threshold_cost <= max(cost_values) <= threshold_cost


# =================================================================================================
#  TEST - Find solution WITHOUT uncertainty - OPTIMAL FIT - fitting procedure
# =================================================================================================
@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [0.0, 0.2, 0.9])
@pytest.mark.parametrize("c_true", [0.25, 0.5, 1.0, 2.0, 4.0, -4.0, -2.0, -1.0, -0.5, -0.25])
@pytest.mark.parametrize("c_noise, tol", [(0.0, 1e-9), (1e-9, 1e-6), (1e-6, 1e-3)])
def test_fit_curve_tailored_accurate(a_true: float, b_true: float, c_true: float, c_noise: float, tol: float):
    """Can we recover the true (a,b,c)-values in the presence of varying levels of noise?"""

    # --- arrange -----------------------------------------
    x_values = compute_x_deltas(dx=0.5, k=5, seed=42)
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)
    for i, x in enumerate(x_values):
        fx_values[i] += c_noise * noise_from_float(x)

    range_a = (0.1, 10.0)
    range_b = (-0.5, 1.0)
    if c_true > 0.0:
        range_c = (0.25, 4.0)
    else:
        range_c = (-4.0, -0.25)

    # --- act ---------------------------------------------
    a_est, b_est, c_est = fit_curve_tailored(x_values, fx_values, range_a, range_b, range_c, reg=0.0, debug_flag=True)

    # --- assert ------------------------------------------
    assert a_est == pytest.approx(a_true, rel=tol, abs=tol)
    assert b_est == pytest.approx(b_true, rel=tol, abs=tol)
    assert c_est == pytest.approx(c_true, rel=tol, abs=tol)


@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [-1.0, 0.5, 1.5])
@pytest.mark.parametrize("c_true", [0.1, 0.5, 1.0, 2.0, 10.0, -5.0, -2.0])
@pytest.mark.parametrize("range_c", [(0.25, 4.0), (-3.0, -0.33)])
@pytest.mark.parametrize("c_noise", [1e-9, 1e-6, 1e-01, 1.0])
def test_fit_curve_tailored_bounds(
    a_true: float, b_true: float, c_true: float, range_c: tuple[float, float], c_noise: float
):
    """Do we always get parameter estimates within bounds?"""

    # --- arrange -----------------------------------------
    x_values = compute_x_deltas(dx=0.5, k=5, seed=42)
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)
    for i, x in enumerate(x_values):
        fx_values[i] += c_noise * noise_from_float(x)

    range_a = (0.1, 10.0)
    range_b = (-0.5, 1.0)

    # --- act ---------------------------------------------
    a_est, b_est, c_est = fit_curve_tailored(x_values, fx_values, range_a, range_b, range_c, reg=0.0)

    # --- assert ------------------------------------------
    assert range_a[0] <= a_est <= range_a[1]
    assert range_b[0] <= b_est <= range_b[1]
    assert range_c[0] <= c_est <= range_c[1]


# =================================================================================================
#  TEST - Find solution WITHOUT uncertainty - OPTIMAL FIT - search directions
# =================================================================================================
@pytest.mark.parametrize("a", [1.0, 2.0])
@pytest.mark.parametrize("b", [-0.5, 0.0, 1.0])
@pytest.mark.parametrize("c", [0.1, 1.0, 10.0, -0.1, -1.0, -10.0])
@pytest.mark.parametrize("step_size", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("method", ["a", "b", "c", "ac", "ba", "bc"])
def test_param_step_bounds(a: float, b: float, c: float, method: str, step_size: float):
    # test if taking maximal step respects parameter bounds

    # --- arrange -----------------------------------------
    range_a = (0.1, 10.0)
    range_b = (-0.5, 1.0)
    if c > 0.0:
        range_c = (0.1, 10.0)
    else:
        range_c = (-10.0, -0.1)

    # --- act ---------------------------------------------
    a_new, b_new, c_new = param_step(a, b, c, method, step_size, range_a, range_b, range_c)

    # --- assert ------------------------------------------
    assert range_a[0] <= a_new <= range_a[1]
    assert range_b[0] <= b_new <= range_b[1]
    assert range_c[0] <= c_new <= range_c[1]
    if step_size == 0.0:
        assert a_new == a
        assert b_new == b
        assert c_new == c


@pytest.mark.parametrize("c", [1.0, -1.0])
def test_param_step_a(c: float):
    # --- arrange -----------------------------------------
    a, b = 1.0, 0.5
    range_a = (0.01, 100.0)
    range_b = (0.0, 1.0)
    if c > 0.0:
        range_c = (0.1, 10.0)
    else:
        range_c = (-10.0, -0.1)

    # --- act ---------------------------------------------
    a_m, b_m, c_m = param_step(a, b, c, "a", -1.0, range_a, range_b, range_c)
    a_p, b_p, c_p = param_step(a, b, c, "a", +1.0, range_a, range_b, range_c)

    # --- assert ------------------------------------------
    assert a_m == pytest.approx(0.1 * a)
    assert a_p == pytest.approx(10 * a)
    assert b_m == b_p == b
    assert c_m == c_p == c


@pytest.mark.parametrize("c", [1.0, -1.0])
def test_param_step_b(c: float):
    # --- arrange -----------------------------------------
    a, b = 1.0, 0.5
    range_a = (0.1, 10.0)
    range_b = (0.0, 1.0)
    if c > 0.0:
        range_c = (0.1, 10.0)
    else:
        range_c = (-10.0, -0.1)

    # --- act ---------------------------------------------
    a_m, b_m, c_m = param_step(a, b, c, "b", -1.0, range_a, range_b, range_c)
    a_p, b_p, c_p = param_step(a, b, c, "b", +1.0, range_a, range_b, range_c)

    # --- assert ------------------------------------------
    assert a_m == a_p == a
    assert b_m == 0.0
    assert b_p == 1.0
    assert c_m == c_p == c


@pytest.mark.parametrize("c", [1.0, -1.0])
def test_param_step_c(c: float):
    # --- arrange -----------------------------------------
    a, b = 1.0, 0.5
    range_a = (0.1, 10.0)
    range_b = (0.0, 1.0)
    if c > 0.0:
        range_c = (0.1, 10.0)
    else:
        range_c = (-10.0, -0.1)

    # --- act ---------------------------------------------
    a_min, b_min, c_min = param_step(a, b, c, "c", -10.0, range_a, range_b, range_c)
    a_m, b_m, c_m = param_step(a, b, c, "c", -1.0, range_a, range_b, range_c)
    a_p, b_p, c_p = param_step(a, b, c, "c", +1.0, range_a, range_b, range_c)
    a_max, b_max, c_max = param_step(a, b, c, "c", +10.0, range_a, range_b, range_c)

    # --- assert ------------------------------------------
    assert a_min == a_m == a_p == a_max == a
    assert b_min == b_m == b_p == b_max == b
    assert c_min == 0.1 * np.sign(c)
    assert c_m == pytest.approx(0.5 * c)
    assert c_p == pytest.approx(2 * c)
    assert c_max == 10.0 * np.sign(c)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0, -1.0])
@pytest.mark.parametrize("step_size", [-0.5, -0.1, 0.2, 0.6])
def test_param_step_ac(c: float, step_size: float):
    # --- arrange -----------------------------------------
    a, b = 1.0, 0.5
    r = 2 * math.sqrt(2)  # this step direction should enforce invariant g(r) - g(1/r) = constant

    range_a = (0.01, 100.0)
    range_b = (0.0, 1.0)
    if c > 0.0:
        range_c = (0.01, 100.0)
    else:
        range_c = (-100.0, -0.0)

    gx_before = fitting_curve(np.array([1 / r, r]), a, b, c)

    a_new_expected, _, _ = param_step(a, b, c, "a", step_size, range_a, range_b, range_c)

    # --- act ---------------------------------------------
    a_new, b_new, c_new = param_step(a, b, c, "ac", step_size, range_a, range_b, range_c)
    gx_after = fitting_curve(np.array([1 / r, r]), a_new, b_new, c_new)

    # --- assert ------------------------------------------
    assert a_new == a_new_expected
    assert b_new == b
    assert c_new != c

    assert (gx_after[1] - gx_after[0]) == pytest.approx(gx_before[1] - gx_before[0], 1e-15)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0, -1.0])
@pytest.mark.parametrize("step_size", [-0.5, -0.1, 0.2, 0.6])
def test_param_step_ba(c: float, step_size: float):
    # --- arrange -----------------------------------------
    a, b = 1.0, 0.5
    r = 2 * math.sqrt(2)  # this step direction should enforce invariant g(r) - g(1/r) = constant

    range_a = (0.01, 100.0)
    range_b = (0.0, 1.0)
    if c > 0.0:
        range_c = (0.01, 100.0)
    else:
        range_c = (-100.0, -0.0)

    gx_before = fitting_curve(np.array([1 / r, r]), a, b, c)

    _, b_new_expected, _ = param_step(a, b, c, "b", step_size, range_a, range_b, range_c)

    # --- act ---------------------------------------------
    a_new, b_new, c_new = param_step(a, b, c, "ba", step_size, range_a, range_b, range_c)
    gx_after = fitting_curve(np.array([1 / r, r]), a_new, b_new, c_new)

    # --- assert ------------------------------------------
    assert a_new != a
    assert b_new == b_new_expected
    assert c_new == c

    assert (gx_after[1] - gx_after[0]) == pytest.approx(gx_before[1] - gx_before[0], 1e-15)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("step_size", [-0.6, -0.2, 0.1, 0.5])
def test_param_step_bc(c: float, step_size: float):
    # --- arrange -----------------------------------------
    a, b = 1.0, 0.5
    r = 2 * math.sqrt(2)  # this step direction should enforce invariant g(r) - g(1/r) = constant

    range_a = (0.01, 100.0)
    range_b = (0.0, 1.0)
    range_c = (0.01, 100.0)

    gx_before = fitting_curve(np.array([1 / r, r]), a, b, c)

    _, b_new_expected, _ = param_step(a, b, c, "b", step_size, range_a, range_b, range_c)

    # --- act ---------------------------------------------
    a_new, b_new, c_new = param_step(a, b, c, "bc", step_size, range_a, range_b, range_c)
    gx_after = fitting_curve(np.array([1 / r, r]), a_new, b_new, c_new)

    # --- assert ------------------------------------------
    assert a_new == a
    assert b_new == b_new_expected
    assert c_new != c

    assert (gx_after[1] - gx_after[0]) == pytest.approx(gx_before[1] - gx_before[0], 1e-15)


# =================================================================================================
#  TEST - Find solution WITHOUT uncertainty - EXACT
# =================================================================================================
@pytest.mark.parametrize(
    "a_true, b_true, c_true",
    [
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (1.0, 0.0, 2.0),
        (2.0, 0.0, 2.0),
        (1.0, 0.0, 3.0),
        (2.0, 0.0, 3.0),
        (1.0, 0.4, 1.0),
        (2.0, 0.9, 1.0),
        (1.0, 0.1, 2.0),
        (2.0, 0.6, 2.0),
        (1.0, 0.0, -1.0),
        (2.0, 0.0, -1.0),
        (1.0, 0.0, -2.0),
        (1.0, 0.0, -3.0),
        (2.0, 0.5, -1.0),
        (2.0, 0.1, -5.0),
    ],
    ids=[
        "linear_1",
        "linear_2",
        "quadratic_1",
        "quadratic_2",
        "cubic_1",
        "cubic_2",
        "linear_step_1",
        "linear_step_2",
        "quadratic_step_1",
        "quadratic_step_2",
        "reciprocal_1",
        "reciprocal_2",
        "reciprocal_3",
        "reciprocal_4",
        "reciprocal_5",
        "reciprocal_6",
    ],
)
def test_fit_tailored_exact_three_points_simple_cases(a_true: float, b_true: float, c_true: float):
    """See if we can reproduce a,b,c in simple cases"""

    # --- arrange -----------------------------------------
    x_values = np.array([0.5, 1.0, 2.0])
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)

    # set range_c such that it has correct sign for capturing exact c
    if c_true > 0:
        range_c = (1e-10, 1e10)
    else:
        range_c = (-1e10, -1e-10)

    # --- act ---------------------------------------------
    a_fit, b_fit, c_fit = fit_curve_exact_three_points(
        fx_05=fx_values[0],
        fx_1=fx_values[1],
        fx_2=fx_values[2],
        range_a=(1e-10, 1e10),
        range_b=(0.0, 1.0),
        range_c=range_c,
    )

    # --- assert ------------------------------------------
    assert a_fit == pytest.approx(a_true, rel=1e-9, abs=1e-9)
    assert b_fit == pytest.approx(b_true, rel=1e-9, abs=1e-9)
    assert c_fit == pytest.approx(c_true, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(
    "a_true, b_true, c_true",
    [
        (0.1, 0.1, 1.0),
        (10.0, 0.1, 1.0),
        (1.0, 0.99, 2.0),
        (1.0, -1.0, 2.0),
        (1.0, 0.1, 0.1),
        (1.0, 0.1, 10.0),
        (0.1, 1.0, 1.0),
        (2.0, -1.0, 10.0),
        (1.0, 0.0, -5.0),
        (1.0, 0.0, -0.2),
        (1.0, -0.1, -5.0),
    ],
    ids=[
        "a_min",
        "a_max",
        "b_min",
        "b_max",
        "c_min",
        "c_max",
        "mixed_ab",
        "mixed_bc",
        "c_neg_1",
        "c_neg_2",
        "c_neg_3",
    ],
)
def test_fit_tailored_exact_three_points_bounds(a_true: float, b_true: float, c_true: float):
    """See if imposed bounds (a in [0.5, 2.0], b in [0.0, 0.9], c in ±[0.5, 2.0]) are always respected"""

    # --- arrange -----------------------------------------
    x_values = np.array([0.5, 1.0, 2.0])
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)

    # set range_c such that it has correct sign for capturing exact c
    if c_true > 0:
        range_c = (0.5, 2.0)
    else:
        range_c = (-2.0, -0.5)

    # --- act ---------------------------------------------
    a_fit, b_fit, c_fit = fit_curve_exact_three_points(
        fx_05=fx_values[0],
        fx_1=fx_values[1],
        fx_2=fx_values[2],
        range_a=(0.5, 2.0),
        range_b=(0.0, 0.9),
        range_c=range_c,
    )

    # --- assert ------------------------------------------
    assert 0.5 <= a_fit <= 2.0
    assert 0.0 <= b_fit <= 0.9
    assert range_c[0] <= c_fit <= range_c[1]


@pytest.mark.parametrize(
    "fx_05, fx_1, fx_2",
    [
        # single zero
        (0.5, 1.0, 0.0),
        (0.5, 0.0, 2.0),
        (0.0, 1.0, 2.0),
        # double zero
        (0.5, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 2.0),
        # all zero
        (0.0, 0.0, 0.0),
        # non-monotonic
        (0.9, 1.0, 0.9),
        (1.1, 1.0, 1.1),
        # decreasing
        (2.0, 1.0, 0.5),
        (1.1, 1.0, 0.5),
        # equal values
        (1.0, 1.0, 2.0),
        (0.5, 1.0, 1.0),
        # single negative
        (-0.5, 1.0, 2.0),
        (0.5, -1.0, 2.0),
        (0.5, 1.0, -2.0),
        # double negative
        (-0.5, -1.0, 2.0),
        (0.5, -1.0, -2.0),
        (-0.5, 1.0, -2.0),
        # all negative
        (-0.5, -1.0, -2.0),
        (-1.0, -1.0, -1.0),
        (-2.0, -1.0, -0.5),
    ],
    ids=[
        "single_zero_1",
        "single_zero_2",
        "single_zero_3",
        "double_zero_1",
        "double_zero_2",
        "double_zero_3",
        "all_zero",
        "non_monotonic_1",
        "non_monotonic_2",
        "decreasing_1",
        "decreasing_2",
        "equal_values_1",
        "equal_values_2",
        "single_negative_1",
        "single_negative_2",
        "single_negative_3",
        "double_negative_1",
        "double_negative_2",
        "double_negative_3",
        "all_negative_1",
        "all_negative_2",
        "all_negative_3",
    ],
)
def test_fit_tailored_exact_ill_conditioning(fx_05: float, fx_1: float, fx_2: float):
    """Check if the function at least gracefully return _some_ result under unexpected circumstances."""

    # --- act & assert ------------------------------------
    a_fit, b_fit, c_fit = fit_curve_exact_three_points(
        fx_05=float(fx_05),
        fx_1=float(fx_1),
        fx_2=float(fx_2),
        range_a=(1e-3, 1e3),
        range_b=(-0.5, 1.5),
        range_c=(1 / 32.0, 32.0),
    )

    # --- assert ------------------------------------------
    assert 1e-3 <= a_fit <= 1e3
    assert -0.5 <= b_fit <= 1.5
    assert 1 / 32.0 <= c_fit <= 32.0


# =================================================================================================
#  TEST - initial_params
# =================================================================================================
@pytest.mark.parametrize("range_a", [(0.1, 10), (2.0, 2.0), (1e-4, 1e4)])
@pytest.mark.parametrize("range_b", [(-0.5, 1.0), (0.0, 1.0), (0.4, 1.0), (-0.2, -0.1)])
@pytest.mark.parametrize("range_c", [(0.1, 10), (-10, -0.1), (2, 5), (-5, -2)])
def test_initial_params(range_a: tuple[float, float], range_b: tuple[float, float], range_c: tuple[float, float]):
    # --- act ---------------------------------------------
    a, b, c = initial_params(range_a, range_b, range_c)

    # --- assert ------------------------------------------
    assert range_a[0] <= a <= range_a[1]
    assert range_b[0] <= b <= range_b[1]
    assert range_c[0] <= c <= range_c[1]

    if range_b[0] <= 0.0 <= range_b[1]:
        assert b == 0.0
    if range_c[0] <= 1.0 <= range_c[1]:
        assert c == 1.0
    elif range_c[0] <= -1.0 <= range_c[1]:
        assert c == -1.0
