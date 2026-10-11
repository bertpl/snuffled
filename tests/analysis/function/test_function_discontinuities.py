import math
from collections.abc import Callable

import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.function import FunctionAnalyser
from snuffled._core.analysis.function.helpers_discontinuous import (
    compute_discontinuity_score_from_intervals,
    merge_samples,
    select_midpoints_to_sample,
)
from snuffled._core.models import FunctionProperty
from snuffled._core.utils.rng_seed import RngSeed


# ==================================================================================================
#  Test functions     (for x-range [-1,1])
# ==================================================================================================
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


def f_constant(x: float) -> float:
    return 1.0  # perfectly flat: dfx_total == 0 used to divide-by-zero; a constant is fully continuous


# ==================================================================================================
#  Main tests
# ==================================================================================================
@pytest.mark.parametrize(
    "fun, min_score, max_score",
    [
        (f_linear, 0.0, 1e-6),
        (f_quad, 0.0, 1e-6),
        (f_exp, 0.0, 1e-3),
        (f_linear_with_step, 0.499, 0.501),
        (f_step, 0.999, 1.0),
        (f_constant, 0.0, 1e-6),
    ],
)
def test_function_analyser_discontinuity(fun: Callable[[float], float], min_score: float, max_score: float):
    # --- arrange ----------------------
    sampler = FunctionSampler(
        n_roots=100,
        fun=fun,
        x_min=-1.0,
        x_max=1.0,
        dx=1e-10,
        rng_seed=RngSeed(42),
        n_fun_samples=1000,
        rel_tol_scale=10.0,
    )
    analyser = FunctionAnalyser(sampler)

    # seed function cache
    _ = sampler.fx_values

    # --- act --------------------------
    discontinuity_score = analyser.extract(FunctionProperty.DISCONTINUOUS)

    # --- assert -----------------------
    assert min_score <= discontinuity_score <= max_score


# ==================================================================================================
#  Helpers
# ==================================================================================================
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
    # --- arrange ----------------------
    x_values = np.array(x_values)
    fx_values = np.array([fun(x) for x in x_values])

    # --- act --------------------------
    score = compute_discontinuity_score_from_intervals(x_values, fx_values, dx_min)

    # --- assert -----------------------
    assert min_score <= score <= max_score


@pytest.mark.parametrize(
    "x_values, fx_values, dx_min, dx_total, n_max, expected_midpoints",
    [
        # the step interval [1, 2] scores highest, followed by the flat intervals from right to left
        ([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0], 1e-3, 3.0, 3, [1.5, 2.5, 0.5]),
        # with n_max = 1, only the midpoint of the highest-scoring interval is returned
        ([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0], 1e-3, 3.0, 1, [1.5]),
        # a constant function gives equal scores, which are ordered right to left
        ([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], 1e-3, 3.0, 2, [2.5, 1.5]),
        # n_max == 0 selects no intervals
        ([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], 1e-3, 3.0, 0, []),
        # an interval no wider than dx_min gets no midpoint, however steep it is
        ([0.0, 1e-4, 1.0], [0.0, 1.0, 1.0], 1e-3, 1.0, 10, [0.5 * (1e-4 + 1.0)]),
    ],
)
def test_select_midpoints_to_sample(
    x_values: list[float],
    fx_values: list[float],
    dx_min: float,
    dx_total: float,
    n_max: int,
    expected_midpoints: list[float],
):
    """select_midpoints_to_sample orders by decreasing score, equal scores right to left, and skips narrow intervals."""
    # --- arrange / act ----------------
    midpoints = select_midpoints_to_sample(np.array(x_values), np.array(fx_values), dx_min, dx_total, n_max)

    # --- assert -----------------------
    assert midpoints.tolist() == expected_midpoints


def test_merge_samples():
    """merge_samples merges new samples in sorted order and drops an x that is already present."""
    # --- arrange ----------------------
    x_values = np.array([0.0, 1.0, 2.0])
    fx_values = np.array([10.0, 11.0, 12.0])
    new_x_values = np.array([1.5, 1.0, 0.5])
    new_fx_values = np.array([11.5, 99.0, 10.5])

    # --- act --------------------------
    merged_x, merged_fx = merge_samples(x_values, fx_values, new_x_values, new_fx_values)

    # --- assert -----------------------
    assert merged_x.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert merged_fx.tolist() == [10.0, 10.5, 11.0, 11.5, 12.0]
