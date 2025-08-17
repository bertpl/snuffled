import math
from typing import Callable

import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function import FunctionAnalyser
from snuffled._core.models import FunctionProperty


# =================================================================================================
#  Test functions     (for x-range [-1,1])
# =================================================================================================
def f_constant(x: float) -> float:
    """Should have an extremely low dynamic range"""
    return 1.0 + (1e-10 * x)


def f_linear(x: float) -> float:
    """Should have a very low dynamic range"""
    return x


def f_exp_1(x: float) -> float:
    """
    Should have a dynamic range around 1e16
    We have f(-0.8) ~= 1e-8     (~=q10)
            f( 0.8)  = 1e8      (~=q90)
    """
    return (10 ** (10 * x)) - (1e-8)


def f_exp_2(x: float) -> float:
    """Should have a dynamic range >1e32"""
    return (10 ** (20 * x)) - (1e-20)


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize(
    "fun, min_score, max_score",
    [
        (f_constant, 0.0, 0.0),
        (f_linear, 0.0, 0.0),
        (f_exp_1, 0.45, 0.55),
        (f_exp_2, 1.0, 1.0),
    ],
)
def test_function_analyser_high_dynamic_range(fun: Callable[[float], float], min_score: float, max_score: float):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=1000)
    analyser = FunctionAnalyser(sampler)

    # --- act ---------------------------------------------
    hdr_score = analyser.extract(FunctionProperty.HIGH_DYNAMIC_RANGE)

    # --- assert ------------------------------------------
    assert min_score <= hdr_score <= max_score
