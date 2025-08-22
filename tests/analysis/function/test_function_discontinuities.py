import math
from typing import Callable

import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function import FunctionAnalyser
from snuffled._core.analysis.function.helpers_discontinuous import compute_discontinuity_score_from_intervals
from snuffled._core.models import FunctionProperty


# =================================================================================================
#  Test functions     (for x-range [-1,1])
# =================================================================================================
def f_linear(x: float) -> float:
    return x


def f_quad(x: float) -> float:
    return x + (x * x) / 10


def f_exp(x: float) -> float:
    return math.exp(10 * x) - 0.5


def f_linear_with_step(x: float) -> float:
    return float(x + np.sign(x))


def f_step(x: float) -> float:
    return float(np.sign(x))


# =================================================================================================
#  Main tests
# =================================================================================================
@pytest.mark.parametrize(
    "fun, min_score, max_score",
    [
        (f_linear, 0.0, 1e-6),
        (f_quad, 0.0, 1e-6),
        (f_exp, 0.0, 1e-3),
        (f_linear_with_step, 0.499, 0.501),
        (f_step, 0.999, 1.0),
    ],
)
def test_function_analyser_discontinuity(fun: Callable[[float], float], min_score: float, max_score: float):
    # --- arrange -----------------------------------------
    function_data = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-10, rel_tol_scale=10.0)
    analyser = FunctionAnalyser(function_data)

    # seed function cache
    _ = function_data.fx_values()

    # --- act ---------------------------------------------
    discontinuity_score = analyser.extract(FunctionProperty.DISCONTINUOUS)

    print(discontinuity_score)

    # --- assert ------------------------------------------
    assert min_score <= discontinuity_score <= max_score


# =================================================================================================
#  Helpers
# =================================================================================================
@pytest.mark.parametrize(
    "fun, x_values, dx_min, min_score, max_score",
    [
        (f_linear, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-9),
        (f_linear, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.0, 1e-9),
        (f_quad, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-6),
        (f_quad, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.0, 1e-6),
        (f_exp, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-3),
        (f_exp, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.0, 1e-3),
        (f_linear_with_step, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-3),
        (f_linear_with_step, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.499, 0.501),
        (f_step, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-3),
        (f_step, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.999, 1.0),
    ],
)
def test_compute_discontinuity_score_from_intervals(
    fun: Callable[[float], float],
    x_values: list[float],
    dx_min: float,
    min_score: float,
    max_score: float,
):
    # --- arrange -----------------------------------------
    x_values = np.array(x_values)
    fx_values = np.array([fun(x) for x in x_values])

    # --- act ---------------------------------------------
    score = compute_discontinuity_score_from_intervals(x_values, fx_values, dx_min)

    print(score)

    # --- assert ------------------------------------------
    assert min_score <= score <= max_score
