import math
from functools import partial
from typing import Callable

import pytest

from snuffled._core.utils.constants import EPS
from snuffled._core.utils.root_finding import determine_root_width, find_odd_root, find_root


# =================================================================================================
#  Test functions      (designed for range [-10, 10])
# =================================================================================================
def f_linear(x: float, root: float) -> float:
    return x - root


def f_quadratic(x: float, root: float) -> float:
    return (x - root) * (x + 11)


def f_exp(x: float, root: float) -> float:
    return math.exp(x) - math.exp(root)


def f_step(x: float, root: float) -> float:
    if x >= root:
        return 1.0
    else:
        return -1.0


def f_near_underflow(x: float) -> float:
    # function that does _not_ suffer from underflow itself, but _will_ cause underflow when computing f(x)*f(y)
    # for x & y inside [-1, 1]
    return (1e-200) * (x - 0.1)


def f_lin_with_wide_zero(x: float, root: float, zero_width: float) -> float:
    if abs(x - root) <= (zero_width / 2):
        return 0.0
    else:
        return x - root


def f_with_wide_even_zero_1(x: float) -> float:
    # true odd root at x=-0.95
    # wide even root in [-0.8, 0.9]
    if x >= 0.9:
        return x - 0.9
    elif -0.8 <= x < 0.9:
        return 0.0
    elif -0.9 <= x < -0.8:
        return 0.9 - x
    else:
        # x < -0.9
        return 2 * (x + 0.95)


def f_with_wide_even_zero_2(x: float) -> float:
    # true odd root at x=0.95
    # wide even root in [-0.9, 0.8]
    return f_with_wide_even_zero_1(-x)


# =================================================================================================
#  Tests - find_odd_root
# =================================================================================================


@pytest.mark.parametrize(
    "fun, expected_odd_root",
    [
        (f_with_wide_even_zero_1, -0.95),
        (f_with_wide_even_zero_2, 0.95),
    ],
)
def test_find_odd_root(fun: Callable[[float], float], expected_odd_root: float):
    # --- act ---------------------------------------------
    root = find_odd_root(fun, x_min=-1.0, x_max=1.0, dx_min=EPS * EPS)

    # --- assert ------------------------------------------
    assert root.x_min <= expected_odd_root <= root.x_max
    assert root.deriv_sign != 0.0


# =================================================================================================
#  Tests - find_root
# =================================================================================================
@pytest.mark.parametrize(
    "x_min, x_max",
    [
        (-1, 1),
        (-3, 3),
        (-10, 10),
    ],
)
@pytest.mark.parametrize(
    "fun, dx_min, x_expected, x_abs_tol",
    [
        (partial(f_linear, root=0.5), EPS * EPS, 0.5, EPS),
        (partial(f_linear, root=-0.01), EPS * EPS, -0.01, 0.01 * EPS),
        (partial(f_linear, root=1e-10), EPS * EPS, 1e-10, 1e-10 * EPS),
        (partial(f_linear, root=1e-50), EPS * EPS, 1e-50, EPS * EPS),
        (partial(f_quadratic, root=0.4), EPS * EPS, 0.4, EPS),
        (partial(f_exp, root=0.6), EPS * EPS, 0.6, EPS),
        (partial(f_exp, root=-0.9), EPS * EPS, -0.9, EPS),
        (partial(f_step, root=-0.8), EPS * EPS, -0.8, EPS),
        (partial(f_step, root=0.1), EPS * EPS, 0.1, EPS),
        (partial(f_step, root=0.99), EPS * EPS, 0.99, EPS),
        (partial(f_step, root=0.0), EPS * EPS, 0.0, EPS * EPS),  # check if dx_min is handled correctly
        (partial(f_step, root=0.0), EPS, 0.0, EPS),  # check if dx_min is handled correctly
        (f_near_underflow, EPS * EPS, 0.1, EPS),  # check if we don't get inaccurate results due to underflow
    ],
)
def test_find_root(
    fun: Callable[[float], float], x_min: float, x_max: float, dx_min: float, x_expected: float, x_abs_tol: float
):
    # --- arrange -----------------------------------------
    x_expected_min = x_expected - x_abs_tol
    x_expected_max = x_expected + x_abs_tol

    # --- act ---------------------------------------------
    root = find_root(fun, x_min, x_max, dx_min)

    # --- assert ------------------------------------------
    assert x_expected_min <= root.x_min <= root.x_max <= x_expected_max


@pytest.mark.parametrize(
    "x_min, x_max, root",
    [
        (3.0, 1.0, 2.0),
        (-1.0, 1.0, -2.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, 2.0),
    ],
)
def test_find_root_value_error(x_min: float, x_max: float, root: float):
    # --- arrange -----------------------------------------
    fun = partial(f_linear, root=root)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = find_root(fun, x_min, x_max, dx_min=EPS)


# =================================================================================================
#  Tests - determine_root_width
# =================================================================================================
@pytest.mark.parametrize("true_root", [-0.9, -0.1, 1 / math.pi, 0.6789])
@pytest.mark.parametrize("true_root_width", [1e-15, 1e-10, 1e-5, 1e-3])
def test_determine_root_width(true_root: float, true_root_width: float):
    # --- arrange -----------------------------------------
    fun = partial(f_lin_with_wide_zero, root=true_root, zero_width=true_root_width)
    x_min = -1.0
    x_max = 1.0

    # --- act ---------------------------------------------
    root = determine_root_width(fun, true_root, x_min, x_max, dx_min=EPS * EPS)

    # --- assert ------------------------------------------
    assert root.x_min <= true_root <= root.x_max
    assert 0.8 * true_root_width <= root.width <= 1.2 * true_root_width


def test_determine_root_width_edge_case():
    """This test set up such that each case has a zero so wide it crosses at least one edge of [x_min, x_max]"""

    # --- arrange -----------------------------------------
    def fun(x: float) -> float:
        # all 0's except at the edge
        if x == -1:
            return -1
        elif x == 1:
            return 1
        else:
            return 0

    x_min = -1.0
    x_max = 1.0

    # --- act ---------------------------------------------
    root = determine_root_width(fun, 0.3, x_min, x_max, dx_min=EPS * EPS)

    # --- assert ------------------------------------------
    assert root.width == 2.0
    assert root.x_min == x_min
    assert root.x_max == x_max
