import math
from functools import partial
from typing import Callable

import pytest

from snuffled._core.utils.constants import EPS
from snuffled._core.utils.root_finding import determine_root_width, find_root, find_root_and_width


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


# =================================================================================================
#  Tests
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
        (partial(f_linear, root=-1.0), EPS * EPS, -1.0, EPS),  # root at the edge if x_min=-1
        (partial(f_linear, root=1.0), EPS * EPS, 1.0, EPS),  # root at the edge if x_max=1
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
    root_min, root_max = find_root(fun, x_min, x_max, dx_min)

    # --- assert ------------------------------------------
    assert x_expected_min <= root_min <= root_max <= x_expected_max
    assert (root_min < root_max) or (fun(root_min) == fun(root_min) == 0.0)
    if root_min < root_max:
        # this means we had to stop due to floating point rounding or dx_min
        # so let's check we didn't overdo it and make it _too_ accurate (which would waste function evaluations)
        assert (root_max - root_min) > (dx_min / 2)


@pytest.mark.parametrize(
    "x_min, x_max",
    [
        (3.0, 1.0),
        (3.0, 5.0),
    ],
)
def test_find_root_value_error(x_min: float, x_max: float):
    # --- arrange -----------------------------------------
    fun = partial(f_linear, root=2.0)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = find_root(fun, x_min, x_max, dx_min=EPS)


@pytest.mark.parametrize("true_root", [-0.9, -0.1, 1 / math.pi, 0.6789])
@pytest.mark.parametrize("true_root_width", [1e-15, 1e-10, 1e-5, 1e-3])
def test_determine_root_width(true_root: float, true_root_width: float):
    # --- arrange -----------------------------------------
    fun = partial(f_lin_with_wide_zero, root=true_root, zero_width=true_root_width)
    x_min = -1.0
    x_max = 1.0

    # --- act ---------------------------------------------
    root_min, root_max = find_root_and_width(fun, x_min, x_max, dx_min=EPS * EPS)

    # --- assert ------------------------------------------
    assert root_min <= true_root <= root_max
    assert 0.8 * true_root_width <= (root_max - root_min) <= 1.2 * true_root_width


@pytest.mark.parametrize(
    "true_root, true_root_width",
    [
        (-0.8, 0.5),
        (0.7, 0.8),
        (-0.99, 0.1),
        (0.93, 0.15),
        (0.123, 3.1),
    ],
)
def test_determine_root_width_edge_cases(true_root: float, true_root_width: float):
    """This test set up such that each case has a zero so wide it crosses at least one edge of [x_min, x_max]"""

    # --- arrange -----------------------------------------
    fun = partial(f_lin_with_wide_zero, root=true_root, zero_width=true_root_width)
    x_min = -1.0
    x_max = 1.0

    # --- act ---------------------------------------------
    root_min, root_max = find_root_and_width(fun, x_min, x_max, dx_min=EPS * EPS)

    # --- assert ------------------------------------------
    assert root_min < root_max
    assert root_min <= true_root <= root_max
    assert root_max - root_min < true_root_width
    assert x_min <= root_min <= x_max
    assert x_min <= root_max <= x_max
    assert (root_min == x_min) or (root_max == x_max)
