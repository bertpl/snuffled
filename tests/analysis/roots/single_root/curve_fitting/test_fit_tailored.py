import numpy as np
import pytest

from snuffled._core.analysis.roots.single_root.curve_fitting import compute_x_deltas, fitting_curve
from snuffled._core.analysis.roots.single_root.curve_fitting._fit_tailored import (
    fit_curve_exact_three_points,
    fit_curve_tailored,
    param_step,
)
from snuffled._core.utils.noise import noise_from_float

# =================================================================================================
#  TEST - Find solution WITH uncertainty
# =================================================================================================
pass


# =================================================================================================
#  TEST - Find solution WITHOUT uncertainty - OPTIMAL FIT - fitting procedure
# =================================================================================================
@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [0.0, 0.2, 0.9])
@pytest.mark.parametrize("c_true", [0.25, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("c_noise, tol", [(0.0, 1e-9), (1e-9, 1e-6), (1e-6, 1e-3)])
def test_fit_curve_tailored_accurate(a_true: float, b_true: float, c_true: float, c_noise: float, tol: float):
    """Can we recover the true (a,b,c)-values in the presence of varying levels of noise?"""

    # --- arrange -----------------------------------------
    x_values = compute_x_deltas(dx=0.5, k=5)
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)
    for i, x in enumerate(x_values):
        fx_values[i] += c_noise * noise_from_float(x)

    range_b = (-0.5, 1.0)
    range_c = (0.25, 4.0)

    # --- act ---------------------------------------------
    a_est, b_est, c_est = fit_curve_tailored(x_values, fx_values, range_b, range_c, reg=0.0)

    # --- assert ------------------------------------------
    assert a_est == pytest.approx(a_true, rel=tol, abs=tol)
    assert b_est == pytest.approx(b_true, rel=tol, abs=tol)
    assert c_est == pytest.approx(c_true, rel=tol, abs=tol)


@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [-1.0, 0.5, 1.5])
@pytest.mark.parametrize("c_true", [0.1, 0.5, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("c_noise", [1e-9, 1e-6, 1e-01, 1.0])
def test_fit_curve_tailored_bounds(a_true: float, b_true: float, c_true: float, c_noise: float):
    """Do we always get parameter estimates within bounds?"""

    # --- arrange -----------------------------------------
    x_values = compute_x_deltas(dx=0.5, k=5)
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)
    for i, x in enumerate(x_values):
        fx_values[i] += c_noise * noise_from_float(x)

    range_b = (-0.5, 1.0)
    range_c = (0.25, 4.0)

    # --- act ---------------------------------------------
    a_est, b_est, c_est = fit_curve_tailored(x_values, fx_values, range_b, range_c, reg=0.0)

    # --- assert ------------------------------------------
    assert a_est > 0
    assert -0.5 <= b_est <= 1.0
    assert 0.25 <= c_est <= 4.0


# =================================================================================================
#  TEST - Find solution WITHOUT uncertainty - OPTIMAL FIT - search directions
# =================================================================================================
@pytest.mark.parametrize("a", [1.0, 2.0])
@pytest.mark.parametrize("b", [-0.5, 0.0, 1.0])
@pytest.mark.parametrize("c", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("step_size", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("method", ["a", "b", "c", "ac", "bc"])
def test_param_step_bounds(a: float, b: float, c: float, method: str, step_size: float):
    # test if taking maximal step respects parameter bounds

    # --- arrange -----------------------------------------
    range_b = (-0.5, 1.0)
    range_c = (0.1, 10.0)

    # --- act ---------------------------------------------
    a_new, b_new, c_new = param_step(a, b, c, method, step_size, range_b, range_c)

    # --- assert ------------------------------------------
    assert 0 < a_new
    assert range_b[0] <= b_new <= range_b[1]
    assert range_c[0] <= c_new <= range_c[1]
    if step_size == 0.0:
        assert a_new == a
        assert b_new == b
        assert c_new == c


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
    ],
)
def test_fit_tailored_exact_three_points_simple_cases(a_true: float, b_true: float, c_true: float):
    """See if we can reproduce a,b,c in simple cases"""

    # --- arrange -----------------------------------------
    x_values = np.array([0.5, 1.0, 2.0])
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)

    # --- act ---------------------------------------------
    a_fit, b_fit, c_fit = fit_curve_exact_three_points(
        fx_05=fx_values[0],
        fx_1=fx_values[1],
        fx_2=fx_values[2],
        range_b=(0.0, 1.0),
        range_c=(1e-10, 1e10),
    )

    # --- assert ------------------------------------------
    assert a_fit == pytest.approx(a_true, rel=1e-9, abs=1e-9)
    assert b_fit == pytest.approx(b_true, rel=1e-9, abs=1e-9)
    assert c_fit == pytest.approx(c_true, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(
    "a_true, b_true, c_true",
    [
        (1.0, 0.0, 0.1),
        (1.0, 0.0, 10.0),
        (1.0, 0.99, 2.0),
        (1.0, -1.0, 2.0),
        (2.0, -1.0, 10.0),
    ],
    ids=[
        "a_min",
        "a_max",
        "b_min",
        "b_max",
        "mixed",
    ],
)
def test_fit_tailored_exact_three_points_bounds(a_true: float, b_true: float, c_true: float):
    """See if imposed bounds (b in [0.0, 0.9], c in [0.5, 2.0]) are always respected"""

    # --- arrange -----------------------------------------
    x_values = np.array([0.5, 1.0, 2.0])
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)

    # --- act ---------------------------------------------
    a_fit, b_fit, c_fit = fit_curve_exact_three_points(
        fx_05=fx_values[0],
        fx_1=fx_values[1],
        fx_2=fx_values[2],
        range_b=(0.0, 0.9),
        range_c=(0.5, 2.0),
    )

    # --- assert ------------------------------------------
    assert 0.0 <= b_fit <= 0.9
    assert 0.5 <= c_fit <= 2.0


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
        range_b=(-0.5, 1.5),
        range_c=(1 / 32.0, 32.0),
    )

    # --- assert ------------------------------------------
    assert -0.5 <= b_fit <= 1.5
    assert 1 / 32.0 <= c_fit <= 32.0
