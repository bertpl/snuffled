import math
from typing import Callable

import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function.function_analyser import FunctionAnalyser
from snuffled._core.analysis.function.helpers_non_monotonic import (
    non_monotonicity_score_n_up_down_flips,
    non_monotonicity_score_up_down_fx,
    non_monotonicity_score_up_down_x,
)
from snuffled._core.models import FunctionProperty
from snuffled._core.utils.noise import noise_from_float
from tests.helpers import is_sorted_with_tolerance


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
#  Tests - Simple cases
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
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=10000, dx=1e-9, rel_tol_scale=10.0)
    analyser = FunctionAnalyser(sampler)
    # --- act ---------------------------------------------
    non_monotonic_score = analyser.extract(FunctionProperty.NON_MONOTONIC)

    print(non_monotonic_score)

    # --- assert ------------------------------------------
    assert min_score - 0.05 <= non_monotonic_score <= max_score + 0.05


# =================================================================================================
#  Tests - Trends
# =================================================================================================
def test_function_analyser_non_monotonic_trend_1():
    """If non-monotonicity becomes more pronounced in y-direction, scores should increase"""

    # --- arrange -----------------------------------------
    def get_f_lin_quad(_c: float) -> Callable[[float], float]:
        # construct function with f(-1)=-1 and f(1)=1, f(0)=c
        def _f_lin_quad(_x: float) -> float:
            return _x + _c * (_x + 1) * (1 - _x)

        return _f_lin_quad

    # --- act ---------------------------------------------
    scores = [
        FunctionAnalyser(
            FunctionSampler(
                fun=get_f_lin_quad(c),
                x_min=-1.0,
                x_max=1.0,
                n_fun_samples=1_000,
                dx=1e-9,
                rel_tol_scale=10.0,
            )
        ).extract(FunctionProperty.NON_MONOTONIC)
        for c in [10**e for e in np.linspace(-1, 10, 20)]
    ]

    print(scores)

    # --- assert ------------------------------------------
    assert is_sorted_with_tolerance(scores, abs_tol=1e-3)
    assert min(scores) == 0.0
    assert max(scores) > 0.60  # 2 of 3 scores will max out in the end


def test_function_analyser_non_monotonic_trend_2():
    """Increasing noise will increase score"""

    # --- arrange -----------------------------------------
    def get_f_exp_noisy(_c: float) -> Callable[[float], float]:
        # construct strongly exponential function with noise of magnitude _c
        def _f_exp_noisy(_x: float) -> float:
            return -2 + math.exp(20 * _x) + _c * noise_from_float(_x)

        return _f_exp_noisy

    # --- act ---------------------------------------------
    scores = [
        FunctionAnalyser(
            FunctionSampler(
                fun=get_f_exp_noisy(c),
                x_min=-1.0,
                x_max=1.0,
                n_fun_samples=1_000,
                dx=1e-9,
                rel_tol_scale=10.0,
            )
        ).extract(FunctionProperty.NON_MONOTONIC)
        for c in [10**e for e in np.linspace(-20, 20, 20)]
    ]

    print(scores)

    # --- assert ------------------------------------------
    assert is_sorted_with_tolerance(scores, abs_tol=1e-3)
    assert min(scores) == 0.0
    assert max(scores) > 0.95


def test_function_analyser_non_monotonic_trend_3():
    """Increasing number of wobbles will increase score"""

    # --- arrange -----------------------------------------
    def get_f_lin_cos(_c: float) -> Callable[[float], float]:
        # construct linear function with smaller amplitude cosine superimposed, with varying frequency
        def _f_lin_cos(_x: float) -> float:
            return 10 * _x + math.cos(_c * _x)

        return _f_lin_cos

    # --- act ---------------------------------------------
    scores = [
        FunctionAnalyser(
            FunctionSampler(
                fun=get_f_lin_cos(c * math.pi),
                x_min=-1.0,
                x_max=1.0,
                n_fun_samples=1_000,
                dx=1e-9,
                rel_tol_scale=10.0,
            )
        ).extract(FunctionProperty.NON_MONOTONIC)
        for c in [10**e for e in np.linspace(0, 10, 11)]
    ]

    print(scores)

    # --- assert ------------------------------------------
    assert is_sorted_with_tolerance(scores, abs_tol=0.1)  # large tol. as n_up_down_flips score is quite stochastic
    assert min(scores) == 0.0
    assert max(scores) > 0.95


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
    score_1 = non_monotonicity_score_up_down_fx(fx_diff)
    score_2 = non_monotonicity_score_up_down_fx(-fx_diff)

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
    score_1 = non_monotonicity_score_up_down_x(diff_fx_signs)
    score_2 = non_monotonicity_score_up_down_x(-diff_fx_signs)

    # --- assert ------------------------------------------
    assert score_1 == score_2
    assert score_1 == pytest.approx(expected_result, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize(
    "diff_fx_signs, expected_result",
    [
        (np.array([-1, 1]), 1.0),
        (np.array([-1, -1, 1]), 1.0),
        (np.array([-1, 0, 1]), 1.0),
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
    score_1 = non_monotonicity_score_n_up_down_flips(diff_fx_signs)
    score_2 = non_monotonicity_score_n_up_down_flips(-diff_fx_signs)

    # --- assert ------------------------------------------
    assert score_1 == score_2
    assert score_1 == pytest.approx(expected_result, rel=1e-15, abs=1e-15)
