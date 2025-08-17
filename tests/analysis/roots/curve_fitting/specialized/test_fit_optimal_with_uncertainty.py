import numpy as np
import pytest

from snuffled._core.analysis.roots.curve_fitting import (
    compute_threshold_cost,
    compute_x_deltas,
    fit_curve,
    fit_curve_with_uncertainty,
    fitting_cost,
    fitting_curve,
)
from snuffled._core.utils.noise import noise_from_float


# =================================================================================================
#  fit_curve_with_uncertainty
# =================================================================================================
@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [0.0, 0.2, 0.9])
@pytest.mark.parametrize("c_true", [0.5, 1.0, 2.0, -2.0, -1.0, -0.5])
@pytest.mark.parametrize("include_opposite_c_range", [False, True])
@pytest.mark.parametrize("c_noise", [0.0, 1e-6, 1e-3, 1.0])
@pytest.mark.parametrize("uncertainty_size", [0.5, 1.0])
@pytest.mark.parametrize("uncertainty_tol", [1e-2, 1e-3])
def test_fit_curve_with_uncertainty(
    a_true: float,
    b_true: float,
    c_true: float,
    include_opposite_c_range: bool,
    c_noise: float,
    uncertainty_size,
    uncertainty_tol,
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
    range_c = (0.25, 4.0)

    # compute reference values to check correctness
    fx_q25, fx_q50, fx_q75 = np.quantile(fx_values, [0.25, 0.50, 0.75])
    _, _, _, cost_opt_pos_c = fit_curve(x_values, fx_values, range_a, range_b, range_c, reg)
    _, _, _, cost_opt_neg_c = fit_curve(x_values, fx_values, range_a, range_b, (-range_c[1], -range_c[0]), reg)
    if include_opposite_c_range:
        cost_opt = min([cost_opt_neg_c, cost_opt_pos_c])
    else:
        cost_opt = cost_opt_pos_c
    threshold_cost = compute_threshold_cost(uncertainty_size, cost_opt, fx_q25, fx_q50, fx_q75)

    # --- act ---------------------------------------------
    a_values, b_values, c_values, cost_values = fit_curve_with_uncertainty(
        x_values,
        fx_values,
        range_a,
        range_b,
        range_c,
        reg,
        include_opposite_c_range=include_opposite_c_range,
        uncertainty_size=uncertainty_size,
        uncertainty_tol=uncertainty_tol,
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
        if include_opposite_c_range:
            assert (range_c[0] <= c <= range_c[1]) or (-range_c[1] <= c <= -range_c[0])
        else:
            assert range_c[0] <= c <= range_c[1]

    # check if cost_values are as expected
    assert (
        min(cost_values) <= cost_opt
    )  # <= not ==, because we can serendipitously a better optimum during uncertainty exploration
    assert max(cost_values) <= threshold_cost
    if (np.sign(range_c[0]) == np.sign(c_true)) or include_opposite_c_range:
        # we expect to find an accurate solution, with reasonably small cost
        # so we expect the uncertainty boundary not to span the entire space, hence we should be able
        # to accurately find the edge
        assert (1 - uncertainty_tol) * threshold_cost <= max(cost_values)

        # we should find an optimal solution with cost dominated by
        #   -> max 1e-2 cost for regularization
        #   -> max 'c_noise' cost due to noise
        assert min(cost_values) <= 1e-2 + c_noise
