import math
from functools import partial
from typing import Callable

import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.utils.constants import EPS


# =================================================================================================
#  FunctionSampler - Basics
# =================================================================================================
def test_function_sampler_f(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    sampler = FunctionSampler(test_fun_quad, x_min, x_max, dx=1e-9, seed=42)

    # --- act & assert ------------------------------------
    assert sampler.f(-1.0) == test_fun_quad(-1.0)
    assert sampler.f(-1.0) == test_fun_quad(-1.0)  # check if cached version also works
    assert sampler.f(0.8) == test_fun_quad(0.8)
    assert sampler.f(1.0) == test_fun_quad(1.0)


def test_function_sampler_f_out_of_range(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    sampler = FunctionSampler(test_fun_quad, x_min, x_max, dx=1e-9, seed=42)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = sampler.f(x_min - 1e-6)

    with pytest.raises(ValueError):
        _ = sampler.f(x_max + 1e-6)


@pytest.mark.parametrize(
    "x_values",
    [
        [],
        [0.123],
        [-0.1, 0.2345, 0.5234],
    ],
)
def test_function_sampler_f_list(test_fun_quad, x_values: list[float]):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(test_fun_quad, x_min=-2.0, x_max=1.0, dx=1e-10, seed=42, n_fun_samples=10)
    expected_result = [test_fun_quad(x) for x in x_values]

    # --- act ---------------------------------------------
    result = sampler.f(x_values)

    # --- assert ------------------------------------------
    assert isinstance(result, list)
    assert len(result) == len(x_values)
    assert np.array_equal(result, expected_result)


def test_function_sampler_f_list_out_of_range(test_fun_quad):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(test_fun_quad, x_min=-2.0, x_max=1.0, dx=1e-10, seed=42, n_fun_samples=10)
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        # out of bounds left
        _ = sampler.f([0.0, -0.5, 0.5, -2.345, 0.8])

    with pytest.raises(ValueError):
        # out of bounds right
        _ = sampler.f([0.0, -0.5, 0.5, 2.345, 0.8])


@pytest.mark.parametrize("dx", [1e-3, 1e-6, 1e-9, 1e-12])
def test_function_sampler_x_values(test_fun_quad, dx: float):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    n = 1000
    sampler = FunctionSampler(test_fun_quad, x_min, x_max, dx=dx, seed=42, n_fun_samples=n)

    # --- act ---------------------------------------------
    x_values = sampler.x_values()
    x_values_2 = sampler.x_values()
    sampler.x_values.cache_clear()
    x_values_3 = sampler.x_values()

    # --- assert ------------------------------------------
    assert np.array_equal(x_values, x_values_2)
    assert np.array_equal(x_values, x_values_3)
    assert len(x_values) == n
    assert np.allclose(x_values[0], x_min)
    assert np.allclose(x_values[-1], x_max)
    assert np.all(np.diff(x_values) > 0)  # values are sorted & strictly different


@pytest.mark.parametrize("warmup_cache", [False, True])
def test_function_sampler_fx_values(test_fun_quad, warmup_cache: bool):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    n = 1000
    dx = 1e-6
    sampler = FunctionSampler(test_fun_quad, x_min, x_max, dx=dx, seed=42, n_fun_samples=n)

    if warmup_cache:
        _ = sampler.f(x_min)
        _ = sampler.f(0.123456)

    # --- act ---------------------------------------------
    x_values = sampler.x_values()
    fx_values = sampler.fx_values()

    # --- assert ------------------------------------------
    for x, fx in zip(x_values, fx_values):
        assert np.isclose(test_fun_quad(x), fx), (
            f"Function value mismatch at x={x}: expected {test_fun_quad(x)}, got {fx}"
        )


@pytest.mark.parametrize(
    "x_before, x_after",
    [
        ([], []),
        ([0.12345], []),
        ([], [0.12345]),
        ([0.12345], [0.12345]),
        ([1.0], []),
        ([], [1.0]),
        ([-2.0, -1.25437], [0.3579, 1.0]),
    ],
)
def test_function_sampler_function_cache(test_fun_quad, x_before: list[float], x_after: list[float]):
    """Tests if .function_cache() returns consistent info after calling .fx_values() and .f(.)"""

    # --- arrange -----------------------------------------
    sampler = FunctionSampler(test_fun_quad, x_min=-2.0, x_max=1.0, dx=1e-10, seed=42, n_fun_samples=10)

    # --- act ---------------------------------------------

    # fetch data before calling x_values(), fx_values()
    x_fx_before = [(x, sampler.f(x)) for x in x_before]

    # call x_values(), fx_values()
    x_values = sampler.x_values()
    fx_values = sampler.fx_values()

    # fetch data after calling x_values(), fx_values()
    x_fx_after = [(x, sampler.f(x)) for x in x_after]

    # get cache contents
    cache_contents = sampler.function_cache()

    # --- assert ------------------------------------------
    assert set(cache_contents) == {(x, test_fun_quad(x)) for x in x_before + list(x_values) + x_after}
    assert set(cache_contents) == {(x, fx) for x, fx in zip(x_values, fx_values)} | set(x_fx_before) | set(x_fx_after)


@pytest.mark.parametrize("n", [100, 1_000, 10_000, 100_000])
def test_function_sampler_robust_estimated_fx_max(n: int):
    # --- arrange -----------------------------------------
    f_linear = lambda x: x
    x_min = -1.0
    x_max = 1.0
    dx = 1e-9
    sampler = FunctionSampler(f_linear, x_min, x_max, dx=dx, seed=42, n_fun_samples=n)

    max_rel_deviation = 1 / math.sqrt(n)

    # --- act ---------------------------------------------
    estimated_max = sampler.robust_estimated_fx_max()

    # --- assert ------------------------------------------
    assert estimated_max == pytest.approx(1.0, rel=max_rel_deviation)


# =================================================================================================
#  FunctionSampler - Tolerances
# =================================================================================================
def test_function_sampler_tol_array_local(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    n = 1000
    dx = 1e-6
    rel_tol_scale = 10.0
    sampler = FunctionSampler(test_fun_quad, x_min, x_max, dx=dx, seed=42, n_fun_samples=n, rel_tol_scale=rel_tol_scale)

    expected_tol_array = abs(sampler.fx_values()) * EPS * rel_tol_scale

    # --- act ---------------------------------------------
    tol_array = sampler.tol_array_local()

    # --- assert ------------------------------------------
    assert np.array_equal(tol_array, expected_tol_array)


def test_function_sampler_tol_array_global(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    n = 1000
    dx = 1e-6
    rel_tol_scale = 10.0
    sampler = FunctionSampler(test_fun_quad, x_min, x_max, dx=dx, seed=42, n_fun_samples=n, rel_tol_scale=rel_tol_scale)

    expected_tol_array = sampler.robust_estimated_fx_max() * EPS * rel_tol_scale * np.ones(n)

    # --- act ---------------------------------------------
    tol_array = sampler.tol_array_global()

    # --- assert ------------------------------------------
    assert np.array_equal(tol_array, expected_tol_array)


# =================================================================================================
#  FunctionSampler - Roots
# =================================================================================================
def f_roots_linear(x: float) -> float:
    return x - 0.1


def f_roots_step(x: float) -> float:
    if x >= 0.1232465789:
        return 1.0
    else:
        return -1.0


def f_roots_sine(x: float, c: float) -> float:
    return math.sin(c * x)


def f_near_underflow(x: float) -> float:
    # function that does _not_ suffer from underflow itself, but _will_ cause underflow when computing f(x)*f(y)
    # for x & y inside [-1, 1]
    return (1e-200) * (x - 0.1)


def f_edge_roots(x: float) -> float:
    """roots at -1 and 1"""
    return x * (x - 1) * (x + 1)


@pytest.mark.parametrize("n_roots", [1, 10, 100])
@pytest.mark.parametrize(
    "fun",
    [
        f_roots_linear,
        f_roots_step,
        partial(f_roots_sine, c=1.0),
        partial(f_roots_sine, c=10.0),
        partial(f_roots_sine, c=1e3),
        f_near_underflow,
    ],
)
def test_function_sampler_roots(fun: Callable[[float], float], n_roots: int):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(
        fun, x_min=-1, x_max=1, dx=1e-10, seed=42, n_fun_samples=1000, n_roots=n_roots, rel_tol_scale=10.0
    )
    candidate_root_intervals, _ = sampler.candidate_root_intervals()

    # --- act ---------------------------------------------
    roots = sampler.roots()

    # --- assert ------------------------------------------
    assert len(roots) == min(n_roots, len(candidate_root_intervals))
    for root_min, root_max in roots:
        f_min, f_max = fun(root_min), fun(root_max)
        assert -1.0 <= root_min <= root_max <= 1.0
        assert root_max - root_min < (4 * EPS)  # all roots are expected to be sharp
        assert (f_min == f_max == 0.0) or (np.sign(f_min) * np.sign(f_max) < 0.0)


@pytest.mark.parametrize(
    "fun, expected_root",
    [
        (f_edge_roots, -1.0),
        (f_edge_roots, 1.0),
    ],
)
def test_function_sampler_roots_edge_cases(fun: Callable[[float], float], expected_root: float):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(
        fun, x_min=-1, x_max=1, dx=1e-10, seed=42, n_fun_samples=1000, n_roots=100, rel_tol_scale=10.0
    )
    candidate_root_intervals, _ = sampler.candidate_root_intervals()

    # --- act ---------------------------------------------
    roots = sampler.roots()

    # --- assert ------------------------------------------
    assert any(root_min <= expected_root <= root_max for root_min, root_max in roots)
