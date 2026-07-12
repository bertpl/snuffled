import time
from collections.abc import Callable

import numpy as np
import pytest

from snuffled import Diagnostic, FunctionProperty, Snuffler
from snuffled._core.models import RootProperty
from tests.helpers import only_with_numba_jit


# =================================================================================================
#  Rootless functions complete extract_all() with all-finite scores (regression R1)
# =================================================================================================
# NOTE: the *flat* rootless functions (constant, all-zeros) additionally need the flat-function
# discontinuity guard, so they are exercised at pipeline level once that lands; here we cover the
# non-flat rootless functions, which the empty-roots fix alone makes crash-free.
@pytest.mark.parametrize(
    "fun",
    [
        lambda x: 1.0 + x * x,  # parabola, no root
        lambda x: x + 5.0,  # monotone, no root in [-1, 1]
    ],
)
def test_snuffler_extract_all_rootless(fun: Callable[[float], float]):
    # --- arrange -----------------------------------------
    snuffler = Snuffler(
        fun=fun, x_min=-1.0, x_max=1.0, dx=1e-10, seed=42, n_fun_samples=1000, n_roots=100, n_root_samples=100
    )

    # --- act ---------------------------------------------
    props = snuffler.extract_all()

    # --- assert ------------------------------------------
    assert np.all(np.isfinite(props.as_array()))


def test_snuffler_supported_properties():
    # --- arrange -----------------------------------------
    snuffler = Snuffler(
        fun=lambda x: x,
        x_min=-1.0,
        x_max=1.0,
        dx=1e-10,
        seed=42,
        n_fun_samples=1000,
        n_roots=100,
        n_root_samples=100,
        rel_tol_scale=10.0,
    )

    # --- act ---------------------------------------------
    supported_properties = snuffler.supported_properties()

    # --- assert ------------------------------------------
    assert set(supported_properties) == set(list(RootProperty) + list(Diagnostic) + list(FunctionProperty))


def test_snuffler_extract_all_smoke():
    """Fast, deterministic structural check of extract_all()/statistics() at tiny sample counts.

    Carries the line coverage of the whole-pipeline tests below, which only run under JIT;
    the parameters are chosen purely for speed, not statistical honesty — no score values
    are asserted here.
    """
    # --- arrange -----------------------------------------
    snuffler = Snuffler(
        fun=lambda x: x,
        x_min=-1.0,
        x_max=1.0,
        dx=1e-10,
        seed=42,
        n_fun_samples=100,
        n_roots=10,
        n_root_samples=10,
        rel_tol_scale=10.0,
    )

    # --- act ---------------------------------------------
    _ = snuffler.extract_all()
    stats = snuffler.statistics()

    # --- assert ------------------------------------------
    assert len(stats) == len(RootProperty) + len(Diagnostic) + len(FunctionProperty)
    assert all(s.t_duration_sec >= 0 for s in stats.values())


@only_with_numba_jit
def test_snuffler_extract_all():
    # --- arrange -----------------------------------------
    snuffler = Snuffler(
        fun=lambda x: x,
        x_min=-1.0,
        x_max=1.0,
        dx=1e-10,
        seed=42,
    )

    # --- act ---------------------------------------------
    _ = snuffler.extract_all()

    # --- assert ------------------------------------------
    # if it doesn't fail at production-default sample counts, that's good enough for this test


@only_with_numba_jit
def test_snuffler_statistics():
    # --- arrange -----------------------------------------
    snuffler = Snuffler(
        fun=lambda x: x,
        x_min=-1.0,
        x_max=1.0,
        dx=1e-10,
        seed=42,
    )

    # --- act ---------------------------------------------
    t_start = time.perf_counter()
    _ = snuffler.extract_all()
    t_end = time.perf_counter()
    stats = snuffler.statistics()

    t_elapsed = t_end - t_start
    t_elapsed_min = (0.99 * t_elapsed) - 1e-3
    t_elapsed_max = (1.01 * t_elapsed) + 1e-3

    # --- assert ------------------------------------------
    assert len(stats) == len(RootProperty) + len(Diagnostic) + len(FunctionProperty)
    assert all(s.t_duration_sec >= 0 for s in stats.values())
    assert t_elapsed_min <= sum([s.t_duration_sec for s in stats.values()]) <= t_elapsed_max
