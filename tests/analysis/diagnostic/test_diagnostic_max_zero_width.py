import math
from functools import partial
from typing import Callable

import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.diagnostic import DiagnosticAnalyser
from snuffled._core.models import Diagnostic
from snuffled._core.utils.constants import EPS


# =================================================================================================
#  Test functions
# =================================================================================================
def f_linear(x: float) -> float:
    return x - 1 / math.pi


def f_cubic(x: float) -> float:
    return x - 2 * (x * x * x) - 0.1


def f_wide_zeroes(x: float, width_left: float, width_right: float) -> float:
    """roots are at -0.5, 0, 0.5"""
    if abs(x + 0.5) <= (width_left / 2):
        return 0.0
    elif abs(x - 0.5) <= (width_right / 2):
        return 0.0
    else:
        return x * (x + 0.5) * (x - 0.5)


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize(
    "fun, max_width_lb, max_width_ub",
    [
        (f_linear, 0.0, 2 * EPS),
        (f_cubic, 0.0, 2 * EPS),
        (partial(f_wide_zeroes, width_left=0, width_right=0), 0.0, 2 * EPS),
        (partial(f_wide_zeroes, width_left=1e-3, width_right=1e-6), 8e-4, 1.2e-3),
        (partial(f_wide_zeroes, width_left=1e-9, width_right=1e-4), 8e-5, 1.2e-4),
        (partial(f_wide_zeroes, width_left=1e-10, width_right=1e-10), 8e-11, 1.2e10),
    ],
)
def test_function_analyser_max_zero_width(fun: Callable[[float], float], max_width_lb: float, max_width_ub: float):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-9, rel_tol_scale=10.0)
    analyser = DiagnosticAnalyser(sampler)

    # --- act ---------------------------------------------
    max_width = analyser.extract(Diagnostic.MAX_ZERO_WIDTH)

    # --- assert ------------------------------------------
    assert max_width_lb <= max_width <= max_width_ub
