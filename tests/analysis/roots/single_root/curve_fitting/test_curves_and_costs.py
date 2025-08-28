import pytest

from snuffled._core.analysis.roots.single_root.curve_fitting._curves_and_costs import compute_threshold_cost


def test_compute_threshold_cost():
    # --- arrange -----------------------------------------
    relative_margin = 1.1122
    optimal_cost = 1.234
    fx_q25, fx_q50, fx_q75 = 1.0, 2.652, 4.1345
    c_opt = 0.999
    c_median = 1.034e-3
    c_range = 1.222e-3

    expected_result = optimal_cost + relative_margin * (
        (c_opt * optimal_cost) + (c_median * abs(fx_q50)) + (c_range * (fx_q75 - fx_q25))
    )

    # --- act ---------------------------------------------
    cost_threshold = compute_threshold_cost(
        relative_margin, optimal_cost, fx_q25, fx_q50, fx_q75, c_opt, c_median, c_range
    )

    # --- assert ------------------------------------------
    assert cost_threshold == pytest.approx(expected_result)
