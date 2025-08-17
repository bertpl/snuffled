import numpy as np
import pytest

from snuffled._core.analysis.roots.curve_fitting import fit_curve_exact_three_points, fitting_curve


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
def test_fit_curve_exact_three_points_simple_cases(a_true: float, b_true: float, c_true: float):
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
def test_fit_curve_exact_three_points_bounds(a_true: float, b_true: float, c_true: float):
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
def test_fit_curve_exact_three_points_ill_conditioning(fx_05: float, fx_1: float, fx_2: float):
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
