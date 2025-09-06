import pytest

from snuffled._core.analysis.roots.curve_fitting import compute_x_deltas, fit_curve, fitting_cost, fitting_curve
from snuffled._core.utils.noise import noise_from_float


# =================================================================================================
#  fit_curve
# =================================================================================================
@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [0.0, 0.2, 0.9])
@pytest.mark.parametrize("c_true", [0.25, 0.5, 1.0, 2.0, 4.0, -4.0, -2.0, -1.0, -0.5, -0.25])
@pytest.mark.parametrize("c_noise, tol", [(0.0, 1e-9), (1e-9, 1e-6), (1e-6, 1e-3)])
def test_fit_curve_accurate(a_true: float, b_true: float, c_true: float, c_noise: float, tol: float):
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
    a_est, b_est, c_est, cost = fit_curve(x_values, fx_values, range_a, range_b, range_c, reg=0.0, debug_flag=True)

    # --- assert ------------------------------------------
    assert a_est == pytest.approx(a_true, rel=tol, abs=tol)
    assert b_est == pytest.approx(b_true, rel=tol, abs=tol)
    assert c_est == pytest.approx(c_true, rel=tol, abs=tol)
    assert cost == fitting_cost(x_values, fx_values, a_est, b_est, c_est, reg=0.0)


@pytest.mark.parametrize("a_true", [1.0, 5.0])
@pytest.mark.parametrize("b_true", [-1.0, 0.5, 1.5])
@pytest.mark.parametrize("c_true", [0.1, 0.5, 1.0, 2.0, 10.0, -5.0, -2.0])
@pytest.mark.parametrize("range_c", [(0.25, 4.0), (-3.0, -0.33)])
@pytest.mark.parametrize("c_noise", [1e-9, 1e-6, 1e-01, 1.0])
def test_fit_curve_bounds(a_true: float, b_true: float, c_true: float, range_c: tuple[float, float], c_noise: float):
    """Do we always get parameter estimates within bounds?"""

    # --- arrange -----------------------------------------
    x_values = compute_x_deltas(dx=0.5, k=5, seed=42)
    fx_values = fitting_curve(x_values, a_true, b_true, c_true)
    for i, x in enumerate(x_values):
        fx_values[i] += c_noise * noise_from_float(x)

    range_a = (0.1, 10.0)
    range_b = (-0.5, 1.0)

    # --- act ---------------------------------------------
    a_est, b_est, c_est, _ = fit_curve(x_values, fx_values, range_a, range_b, range_c, reg=0.0)

    # --- assert ------------------------------------------
    assert range_a[0] <= a_est <= range_a[1]
    assert range_b[0] <= b_est <= range_b[1]
    assert range_c[0] <= c_est <= range_c[1]
