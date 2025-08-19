import math
from typing import Callable

import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function.analyser import (
    FunctionAnalyser,
    _non_monotonicity_score_n_up_down_flips,
    _non_monotonicity_score_up_down_fx,
    _non_monotonicity_score_up_down_x,
)
from snuffled._core.models import FunctionProperty


# =================================================================================================
#  Test functions     (for x-range [-1,1])
# =================================================================================================
def f_quad(x: float) -> float:
    """
    score_up_down_fx       -->   1.0
    score_up_down_x        -->   1.0
    score_n_up_down_signs  -->  ~0.0
    """
    return (x * x) - 0.1


def f_lin_quad(x: float) -> float:
    """
    score_up_down_fx       -->   0.25/2.25 = 1/9
    score_up_down_x        -->    0.5/1.5  = 1/3
    score_n_up_down_signs  -->              ~0.0
    """
    return x + (x * x) - 0.1


def f_sin(x: float) -> float:
    """
    score_up_down_fx       -->   1.0
    score_up_down_x        -->   1.0
    score_n_up_down_signs  -->  ~0.0
    """
    return math.sin(x * math.pi)


# =================================================================================================
#  Test - Main method
# =================================================================================================
@pytest.mark.parametrize(
    "fun, min_score, max_score",
    [
        (f_quad, 2 / 3, 2 / 3),
        (f_lin_quad, 4 / 27, 4 / 27),
        (f_sin, 2 / 3, 2 / 3),
    ],
)
def test_function_analyser_non_monotonic(fun: Callable[[float], float], min_score: float, max_score: float):
    # --- arrange -----------------------------------------
    function_data = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-9, rel_tol_scale=10.0)
    analyser = FunctionAnalyser(function_data)
    # --- act ---------------------------------------------
    function_props = analyser.analyse()
    non_monotonic_score = function_props[FunctionProperty.NON_MONOTONIC]

    print(non_monotonic_score)

    # --- assert ------------------------------------------
    assert min_score - 0.05 <= non_monotonic_score <= max_score + 0.05


# =================================================================================================
#  Tests - Helpers
# =================================================================================================
@pytest.mark.parametrize(
    "fx_diff, expected_result",
    [
        (np.array([-10, 10]), 1.0),
        (np.array([-1, -1, 1, 1]), 1.0),
        (np.array([-1, 0.25, 0.25, 0.25, 0.25]), 1.0),
        (np.array([-1, 1, -1, -1, -1]), 0.25),
        (np.array([-1, 10, -1, -1, -1]), 0.4),
        (np.array([1, 1, 1, 1, 0, 0, 0]), 0.0),
        (np.array([0, 0, 0, 0, 0, 0, 0]), 0.0),
    ],
)
def test_score_total_up_down_fx(fx_diff: np.ndarray, expected_result: float):
    # --- act ---------------------------------------------
    score_1 = _non_monotonicity_score_up_down_fx(fx_diff)
    score_2 = _non_monotonicity_score_up_down_fx(-fx_diff)

    # --- assert ------------------------------------------
    assert score_1 == score_2
    assert score_1 == pytest.approx(expected_result, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize(
    "diff_fx_signs, expected_result",
    [
        (np.array([-1, 1]), 1.0),
        (np.array([-1, -1, 1, 1]), 1.0),
        (np.array([-1, 0.25, 0.25, 0.25, 0.25]), 1.0),
        (np.array([-1, 1, -1, -1, -1]), 0.25),
        (np.array([1, 1, 1, 1, 0, 0, 0]), 0.0),
        (np.array([0, 0, 0, 0, 0, 0, 0]), 0.0),
    ],
)
def test_score_total_up_down_x(diff_fx_signs: np.ndarray, expected_result: float):
    # --- act ---------------------------------------------
    score_1 = _non_monotonicity_score_up_down_x(diff_fx_signs)
    score_2 = _non_monotonicity_score_up_down_x(-diff_fx_signs)

    # --- assert ------------------------------------------
    assert score_1 == score_2
    assert score_1 == pytest.approx(expected_result, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize(
    "diff_fx_signs, expected_result",
    [
        (np.array([-1, 1]), 1.0),
        (np.array([-1, -1, 1]), 1.0),
        (np.array([-1, -1, 1, 1, 1]), 0.5),
        (np.array([-0.5, 0.5, 0.5]), 0.5),
        (np.array([-1.0, 0.5, 0.5]), 0.5),
        (np.array([-1.0, 0.2, 0.5]), 0.5),
        (np.array([-1.0, 0.1, 0.2]), 0.2),
        (np.array([-1.0, 0.01, -1.0]), 0.02),
        (np.array([1, 1, 1, 1, 1]), 0.0),
        (np.array([1, 1, 1, 1, 0]), 0.0),
        (np.array([0, 0, 0, 0, 0, 0, 0]), 0.0),
    ],
)
def test_score_n_up_down_flips(diff_fx_signs: np.ndarray, expected_result: float):
    # --- act ---------------------------------------------
    score_1 = _non_monotonicity_score_n_up_down_flips(diff_fx_signs)
    score_2 = _non_monotonicity_score_n_up_down_flips(-diff_fx_signs)

    # --- assert ------------------------------------------
    assert score_1 == score_2
    assert score_1 == pytest.approx(expected_result, rel=1e-15, abs=1e-15)
