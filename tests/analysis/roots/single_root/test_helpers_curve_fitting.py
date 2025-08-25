import numpy as np
import pytest

from snuffled._core.analysis.roots.single_root.helpers_curve_fitting import fit_curve_brute_force
from snuffled._core.utils.noise import noise_from_float


# =================================================================================================
#  fit_curve_brute_force
# =================================================================================================
@pytest.mark.parametrize("c_noise", [1e-9, 1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("true_c_exp", [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0])
def test_fit_curve_brute_force_c_exp(true_c_exp: float, c_noise: float):
    """Check if for simple cases we can reliably reverse-engineer c_exp"""

    # --- arrange -----------------------------------------
    x_values = np.linspace(0.5, 1.5, 20)
    fx_values = np.array([(x**true_c_exp) + (c_noise * noise_from_float(x)) for x in x_values])

    # --- act ---------------------------------------------
    c_exp_values, c_step_values, cost_values = fit_curve_brute_force(
        x=x_values,
        fx=fx_values,
        range_c_exp=(0.1, 10),
        range_c_step=(-0.5, 1.5),
        exp_sign=1.0,
        n_grid=201,
        c_reg=1e-3,
        tol_c0=0.0,
        tol_c1=2.0,
    )

    print(f"c_exp  : {min(c_exp_values)} -> {max(c_exp_values)}")
    print(f"c_step : {min(c_step_values)} -> {max(c_step_values)}")

    # --- assert ------------------------------------------
    # NOTE: we need to add some tolerances, since we search over a fixed grid
    #       that might not even contain the true value.
    assert 0.99 * min(c_exp_values) <= true_c_exp <= 1.01 * max(c_exp_values)
    assert min(c_step_values) - 0.01 <= 0.0 <= max(c_step_values) + 0.01


@pytest.mark.parametrize("c_noise", [1e-9, 1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("true_c_step", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_fit_curve_brute_force_c_step(true_c_step: float, c_noise: float):
    """Check if for simple cases we can reliably reverse-engineer c_step"""

    # --- arrange -----------------------------------------
    x_values = np.linspace(0.5, 1.5, 20)
    fx_values = np.array([true_c_step + (1 - true_c_step) * x + (c_noise * noise_from_float(x)) for x in x_values])

    # --- act ---------------------------------------------
    c_exp_values, c_step_values, cost_values = fit_curve_brute_force(
        x=x_values,
        fx=fx_values,
        range_c_exp=(0.1, 10),
        range_c_step=(-0.5, 1.5),
        exp_sign=1.0,
        n_grid=201,
        c_reg=1e-3,
        tol_c0=0.0,
        tol_c1=2.0,
    )

    print(f"c_exp  : {min(c_exp_values)} -> {max(c_exp_values)}")
    print(f"c_step : {min(c_step_values)} -> {max(c_step_values)}")

    # --- assert ------------------------------------------
    # NOTE: we need to add some tolerances, since we search over a fixed grid
    #       that might not even contain the true value.
    assert 0.99 * min(c_exp_values) <= 1.0 <= 1.01 * max(c_exp_values)
    assert min(c_step_values) - 0.01 <= true_c_step <= max(c_step_values) + 0.01
