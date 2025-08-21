import math

import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis._function_sampler import smoothen_fx_abs_tol, smoothen_fx_rel_tol
from snuffled._core.utils.constants import EPS


# =================================================================================================
#  FunctionSampler - Basics
# =================================================================================================
def test_function_sampler_f(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    fun_sampler = FunctionSampler(test_fun_quad, x_min, x_max)

    # --- act & assert ------------------------------------
    assert fun_sampler.f(-1.0) == test_fun_quad(-1.0)
    assert fun_sampler.f(-1.0) == test_fun_quad(-1.0)  # check if cached version also works
    assert fun_sampler.f(0.8) == test_fun_quad(0.8)
    assert fun_sampler.f(1.0) == test_fun_quad(1.0)


def test_function_sampler_f_out_of_range(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    fun_sampler = FunctionSampler(test_fun_quad, x_min, x_max)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = fun_sampler.f(x_min - 1e-6)

    with pytest.raises(ValueError):
        _ = fun_sampler.f(x_max + 1e-6)


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
    fun_sampler = FunctionSampler(test_fun_quad, x_min=-2.0, x_max=1.0, n_fun_samples=10, dx=1e-10)
    expected_result = [test_fun_quad(x) for x in x_values]

    # --- act ---------------------------------------------
    result = fun_sampler.f(x_values)

    # --- assert ------------------------------------------
    assert isinstance(result, list)
    assert len(result) == len(x_values)
    assert np.array_equal(result, expected_result)


def test_function_sampler_f_list_out_of_range(test_fun_quad):
    # --- arrange -----------------------------------------
    fun_sampler = FunctionSampler(test_fun_quad, x_min=-2.0, x_max=1.0, n_fun_samples=10, dx=1e-10)
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        # out of bounds left
        _ = fun_sampler.f([0.0, -0.5, 0.5, -2.345, 0.8])

    with pytest.raises(ValueError):
        # out of bounds right
        _ = fun_sampler.f([0.0, -0.5, 0.5, 2.345, 0.8])


@pytest.mark.parametrize("dx", [1e-3, 1e-6, 1e-9, 1e-12])
def test_function_sampler_x_values(test_fun_quad, dx: float):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    n = 1000
    fun_sampler = FunctionSampler(test_fun_quad, x_min, x_max, n_fun_samples=n, dx=dx)

    # --- act ---------------------------------------------
    x_values = fun_sampler.x_values()
    x_values_2 = fun_sampler.x_values()
    fun_sampler.x_values.cache_clear()
    x_values_3 = fun_sampler.x_values()

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
    fun_sampler = FunctionSampler(test_fun_quad, x_min, x_max, n_fun_samples=n, dx=dx)

    if warmup_cache:
        _ = fun_sampler.f(x_min)
        _ = fun_sampler.f(0.123456)

    # --- act ---------------------------------------------
    x_values = fun_sampler.x_values()
    fx_values = fun_sampler.fx_values()

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
    fun_sampler = FunctionSampler(test_fun_quad, x_min=-2.0, x_max=1.0, n_fun_samples=10, dx=1e-10)

    # --- act ---------------------------------------------

    # fetch data before calling x_values(), fx_values()
    x_fx_before = [(x, fun_sampler.f(x)) for x in x_before]

    # call x_values(), fx_values()
    x_values = fun_sampler.x_values()
    fx_values = fun_sampler.fx_values()

    # fetch data after calling x_values(), fx_values()
    x_fx_after = [(x, fun_sampler.f(x)) for x in x_after]

    # get cache contents
    cache_contents = fun_sampler.function_cache()

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
    fun_sampler = FunctionSampler(f_linear, x_min, x_max, n_fun_samples=n, dx=dx)

    max_rel_deviation = 1 / math.sqrt(n)

    # --- act ---------------------------------------------
    estimated_max = fun_sampler.robust_estimated_fx_max()

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
    fun_sampler = FunctionSampler(test_fun_quad, x_min, x_max, n_fun_samples=n, dx=dx, rel_tol_scale=rel_tol_scale)

    expected_tol_array = abs(fun_sampler.fx_values()) * EPS * rel_tol_scale

    # --- act ---------------------------------------------
    tol_array = fun_sampler.tol_array_local()

    # --- assert ------------------------------------------
    assert np.array_equal(tol_array, expected_tol_array)


def test_function_sampler_tol_array_global(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    n = 1000
    dx = 1e-6
    rel_tol_scale = 10.0
    fun_sampler = FunctionSampler(test_fun_quad, x_min, x_max, n_fun_samples=n, dx=dx, rel_tol_scale=rel_tol_scale)

    expected_tol_array = fun_sampler.robust_estimated_fx_max() * EPS * rel_tol_scale * np.ones(n)

    # --- act ---------------------------------------------
    tol_array = fun_sampler.tol_array_global()

    # --- assert ------------------------------------------
    assert np.array_equal(tol_array, expected_tol_array)


def test_function_sampler_fx_values_smoothed(test_fun_quad):
    # --- arrange -----------------------------------------
    x_min, x_max = -2.0, 1.0
    n = 1000
    dx = 1e-6
    rel_tol_scale = 1e10  # influences smoothing; chosen large to see an effect with this simple, well-behaved function
    fun_sampler = FunctionSampler(test_fun_quad, x_min, x_max, n_fun_samples=n, dx=dx, rel_tol_scale=rel_tol_scale)

    abs_tol = fun_sampler.rel_tol * fun_sampler.fx_quantile(0.9, absolute=True)
    rel_tol = fun_sampler.rel_tol

    # --- act ---------------------------------------------
    fx_raw = fun_sampler.fx_values(smoothing="")
    fx_smooth_abs = fun_sampler.fx_values(smoothing="absolute")
    fx_smooth_rel = fun_sampler.fx_values(smoothing="relative")

    # --- assert ------------------------------------------
    assert not np.array_equal(fx_raw, fx_smooth_abs), "test not well set up; increase rel_tol_scale"
    assert not np.array_equal(fx_raw, fx_smooth_rel), "test not well set up; increase rel_tol_scale"
    assert not np.array_equal(fx_smooth_abs, fx_smooth_rel), "test not well set up; increase rel_tol_scale"
    assert np.array_equal(fx_smooth_abs, smoothen_fx_abs_tol(fx_raw, abs_tol=abs_tol))
    assert np.array_equal(fx_smooth_rel, smoothen_fx_rel_tol(fx_raw, rel_tol=rel_tol))
