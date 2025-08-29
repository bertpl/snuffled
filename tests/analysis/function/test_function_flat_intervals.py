from typing import Callable

import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function import FunctionAnalyser
from snuffled._core.models import FunctionProperty
from snuffled._core.utils.noise import noise_from_float
from tests.helpers import is_sorted_with_tolerance


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
        (f_exp_2, 0.40, 0.55),
        (f_relu, 0.45, 0.55),
        (f_rounded_2, 0.90, 1.00),
        (f_constant, 0.99, 1.00),
    ],
)
def test_function_analyser_flat_intervals(fun: Callable[[float], float], min_score: float, max_score: float):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=10000, rel_tol_scale=10.0)
    analyser = FunctionAnalyser(sampler)

    # --- act ---------------------------------------------
    flat_score = analyser.extract(FunctionProperty.FLAT_INTERVALS)

    print(flat_score)

    # --- assert ------------------------------------------
    assert min_score <= flat_score <= max_score


def test_function_analyser_flat_intervals_trend_1():
    """
    For wider fully flat intervals, the score should go up.
    """

    # --- arrange -----------------------------------------
    def get_f_relu(_x0: float) -> Callable[[float], float]:
        # construct function that is fully flat for x <= x0
        def _f_relu(_x: float) -> float:
            return max(0.0, _x - _x0) - 0.5

        return _f_relu

    # --- act ---------------------------------------------
    scores = [
        FunctionAnalyser(
            FunctionSampler(
                fun=get_f_relu(x0),
                x_min=-1.0,
                x_max=1.0,
                dx=1e-9,
                seed=42,
                n_fun_samples=10_000,
                rel_tol_scale=10.0,
            )
        ).extract(FunctionProperty.FLAT_INTERVALS)
        for x0 in np.linspace(-0.95, 0.95, 100)
    ]

    # --- assert ------------------------------------------
    assert is_sorted_with_tolerance(scores, abs_tol=1e-3), "Scores should be non-decreasing"
    assert 0.0 < min(scores) < 0.1
    assert 0.9 < max(scores) < 1.0
    assert max(np.diff(scores)) < 0.05, "change should be gradual"


def test_function_analyser_flat_intervals_trend_2():
    """
    For gradually flatter function ranges, the score should go up.
    """

    # --- arrange -----------------------------------------
    def get_f_exp(_c: float) -> Callable[[float], float]:
        # construct function that is more and more flat for x<0 wrt f(1.0)
        def _f_exp(_x: float) -> float:
            # we expect 10**_c > 1.0
            return (10 ** (_c * _x)) - 1.0

        return _f_exp

    # --- act ---------------------------------------------
    scores = [
        FunctionAnalyser(
            FunctionSampler(
                fun=get_f_exp(c),
                x_min=-1.0,
                x_max=1.0,
                dx=1e-9,
                seed=42,
                n_fun_samples=10_000,
                rel_tol_scale=10.0,
            )
        ).extract(FunctionProperty.FLAT_INTERVALS)
        for c in np.linspace(1, 20, 100)
    ]

    # --- assert ------------------------------------------
    assert is_sorted_with_tolerance(scores, abs_tol=1e-3), "Scores should be non-decreasing"
    assert min(scores) < 1e-3
    assert max(scores) > 0.4
    assert max(np.diff(scores)) < 0.05, "change should be gradual"


def test_function_analyser_flat_intervals_trend_3():
    """
    Adding noise of increasing magnitude to flat areas should gradually decrease flatness score
    """

    # --- arrange -----------------------------------------
    def get_f_noisy_exp(_c: float) -> Callable[[float], float]:
        # construct strongly exponential function with noise of magnitude c
        def _f_noisy_exp(_x: float) -> float:
            return (10 ** (20 * _x)) - 1.0 + (_c * noise_from_float(_x))

        return _f_noisy_exp

    # --- act ---------------------------------------------
    c_values = [10**e for e in np.linspace(10, -20, 100)]
    scores = [
        FunctionAnalyser(
            FunctionSampler(
                fun=get_f_noisy_exp(_c=c),
                x_min=-1.0,
                x_max=1.0,
                dx=1e-9,
                seed=42,
                n_fun_samples=10_000,
                rel_tol_scale=10.0,
            )
        ).extract(FunctionProperty.FLAT_INTERVALS)
        for c in c_values
    ]

    # --- assert ------------------------------------------
    assert is_sorted_with_tolerance(scores, abs_tol=1e-3), "Scores should be non-decreasing"
    assert min(scores) < 1e-2
    assert max(scores) > 0.45
    assert max(abs(np.diff(scores))) < 0.1, "change should be gradual"
