import math
from functools import partial
from typing import Callable

import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function import FunctionAnalyser
from snuffled._core.models import FunctionProperty


# =================================================================================================
#  Test functions      (for range [-1,1])
# =================================================================================================
def f_linear(x: float) -> float:
    return x - 0.123456789


def f_cubic(x: float) -> float:
    return (x - 0.123456789) - 2 * (x * x * x)


def f_sine(x: float, n_roots: int) -> float:
    return math.sin(1.0 + (x * n_roots * 0.5 * math.pi))


# =================================================================================================
#  Main tests
# =================================================================================================
@pytest.mark.parametrize(
    "fun, min_score, max_score",
    [
        (f_linear, 0.0, 1e-3),
        (f_cubic, 0.05, 0.20),
        (partial(f_sine, n_roots=1), 0.0, 1e-3),
        (partial(f_sine, n_roots=3), 0.05, 0.20),
        (partial(f_sine, n_roots=1000), 0.25, 0.75),
        (partial(f_sine, n_roots=1e9), 0.95, 1.00),
    ],
)
def test_function_analyser_many_zeroes(fun: Callable[[float], float], min_score: float, max_score: float):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-6)
    analyser = FunctionAnalyser(sampler)

    # --- act ---------------------------------------------
    many_zeroes_score = analyser.extract(FunctionProperty.MANY_ZEROES)

    # --- assert ------------------------------------------
    assert min_score <= many_zeroes_score <= max_score
