import math

import numpy as np
import pytest

from snuffled._core.analysis.roots.curve_fitting import fitting_curve
from snuffled._core.analysis.roots.curve_fitting.specialized._helpers import initialize_params, param_step


# =================================================================================================
#  initial_params
# =================================================================================================
@pytest.mark.parametrize("range_a", [(0.1, 10), (2.0, 2.0), (1e-4, 1e4)])
@pytest.mark.parametrize("range_b", [(-0.5, 1.0), (0.0, 1.0), (0.4, 1.0), (-0.2, -0.1)])
@pytest.mark.parametrize("range_c", [(0.1, 10), (-10, -0.1), (2, 5), (-5, -2)])
def test_initial_params(range_a: tuple[float, float], range_b: tuple[float, float], range_c: tuple[float, float]):
    # --- act ---------------------------------------------
    a, b, c = initialize_params(range_a, range_b, range_c)

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


# =================================================================================================
#  param_step
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
