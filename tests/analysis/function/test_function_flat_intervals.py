import math
from typing import Callable

import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function import FunctionAnalyser
from snuffled._core.models import FunctionProperty


# =================================================================================================
#  Test functions     (for x-range [-1,1])
# =================================================================================================
def f_linear(x: float) -> float:
    """Should be found to be 0% flat"""
    return x


def f_rounded_1(x: float) -> float:
    """Should be found to be non-0% flat"""
    return round(x * 1e6) / 1e-6


def f_rounded_2(x: float) -> float:
    """Should be found to be >90% flat"""
    return round(x * 10) / 10


def f_constant(x: float) -> float:
    """Should be found to be fully flat"""
    return 1.0


def f_relu(x: float) -> float:
    """Should be found to be ~50% flat"""
    return max(0.0, x) - 0.5


def f_exp_1(x: float) -> float:
    """Should be found to be ~30% flat"""
    return 10 ** (15 * x)


def f_exp_2(x: float) -> float:
    """Should be found to be ~40-50% flat (rel: x ~< -0.1, abs: x ~< 0)"""
    return (10 ** (20 * x)) - 1.0


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize(
    "fun, min_score, max_score",
    [
        (f_linear, 0.00, 0.01),
        (f_rounded_1, 0.05, 1.00),
        (f_exp_1, 0.25, 0.35),
        (f_exp_2, 0.40, 0.50),
        (f_relu, 0.45, 0.55),
        (f_rounded_2, 0.90, 1.00),
        (f_constant, 0.99, 1.00),
    ],
)
def test_function_analyser_flat_intervals(fun: Callable[[float], float], min_score: float, max_score: float):
    # --- arrange -----------------------------------------
    function_data = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-9, rel_tol_scale=10.0)
    analyser = FunctionAnalyser(function_data)

    # --- act ---------------------------------------------
    function_props = analyser.analyse()
    flat_score = function_props[FunctionProperty.FLAT_INTERVALS]

    print(flat_score)

    # --- assert ------------------------------------------
    assert min_score <= flat_score <= max_score
