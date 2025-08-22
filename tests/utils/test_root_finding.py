import math
from functools import partial
from typing import Callable

import pytest

from snuffled._core.utils.constants import EPS
from snuffled._core.utils.root_finding import find_root


# =================================================================================================
#  Test functions      (designed for range [-10, 10])
# =================================================================================================
def f_linear(x: float, root: float) -> float:
    return x - root


def f_quadratic(x: float, root: float):
    return (x - root) * (x + 11)


def f_exp(x: float, root: float):
    return math.exp(x) - math.exp(root)


def f_step(x: float, root: float):
    if x >= root:
        return 1.0
    else:
        return -1.0


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
    "fun, x_expected, x_abs_tol",
    [
        (partial(f_linear, root=0.5), 0.5, EPS),
        (partial(f_linear, root=-0.01), -0.01, 0.01 * EPS),
        (partial(f_linear, root=1e-10), 1e-10, 1e-10 * EPS),
        (partial(f_linear, root=1e-50), 1e-50, EPS * EPS),
        (partial(f_quadratic, root=0.4), 0.4, EPS),
        (partial(f_exp, root=0.6), 0.6, EPS),
        (partial(f_exp, root=-0.9), -0.9, EPS),
        (partial(f_step, root=-0.8), -0.8, EPS),
        (partial(f_step, root=0.1), 0.1, EPS),
        (partial(f_step, root=0.99), 0.99, EPS),
    ],
)
def test_find_root(fun: Callable[[float], float], x_min: float, x_max: float, x_expected: float, x_abs_tol: float):
    # --- arrange -----------------------------------------
    x_expected_min = x_expected - x_abs_tol
    x_expected_max = x_expected + x_abs_tol

    # --- act ---------------------------------------------
    x = find_root(fun, x_min, x_max)

    # --- assert ------------------------------------------
    assert x_expected_min <= x <= x_expected_max


@pytest.mark.parametrize(
    "fun, x_min, x_max, x_expected",
    [
        (partial(f_linear, root=-1.0), -1.0, 1.0, -1.0),
        (partial(f_linear, root=1.0), -1.0, 1.0, 1.0),
        (partial(f_exp, root=-2.0), -2.0, 1.0, -2.0),
        (partial(f_exp, root=3.0), -1.0, 3.0, 3.0),
    ],
)
def test_find_root_edge_cases(fun: Callable[[float], float], x_min: float, x_max: float, x_expected: float):
    # --- act ---------------------------------------------
    x = find_root(fun, x_min, x_max)

    # --- assert ------------------------------------------
    assert x == x_expected


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
        _ = find_root(fun, x_min, x_max)
