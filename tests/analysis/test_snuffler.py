import time

from snuffled import Diagnostic, FunctionProperty, Snuffler
from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.roots import RootsAnalyser
from snuffled._core.models import RootProperty


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


def test_snuffler_extract_all():
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
    _ = snuffler.extract_all()

    # --- assert ------------------------------------------
    pass  # if it doesn't fail, that's good enough for this test


def test_snuffler_statistics():
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
