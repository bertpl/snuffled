import time

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.diagnostic import DiagnosticAnalyser
from snuffled._core.models import Diagnostic


def test_diagnostic_analyser_supported_properties():
    # --- arrange -----------------------------------------
    function_data = FunctionSampler(
        fun=lambda x: x, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-10, rel_tol_scale=10.0
    )
    analyser = DiagnosticAnalyser(function_data)

    # --- act ---------------------------------------------
    supported_properties = analyser.supported_properties()

    # --- assert ------------------------------------------
    assert set(supported_properties) == set(Diagnostic)


def test_diagnostic_analyser_extract_all():
    # --- arrange -----------------------------------------
    function_data = FunctionSampler(
        fun=lambda x: x, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-10, rel_tol_scale=10.0
    )
    analyser = DiagnosticAnalyser(function_data)

    # --- act ---------------------------------------------
    _ = analyser.extract_all()

    # --- assert ------------------------------------------
    pass  # if it doesn't fail, that's good enough for this test


def test_diagnostic_analyser_statistics():
    # --- arrange -----------------------------------------
    function_data = FunctionSampler(
        fun=lambda x: x, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-10, rel_tol_scale=10.0
    )
    analyser = DiagnosticAnalyser(function_data)

    # --- act ---------------------------------------------
    t_start = time.perf_counter()
    _ = analyser.extract_all()
    t_end = time.perf_counter()
    stats = analyser.statistics()

    t_elapsed = t_end - t_start
    t_elapsed_min = (0.99 * t_elapsed) - 1e-3
    t_elapsed_max = (1.01 * t_elapsed) + 1e-3

    # --- assert ------------------------------------------
    assert len(stats) == len(Diagnostic)
    assert all(s.t_duration_sec >= 0 for s in stats.values())
    assert t_elapsed_min <= sum([s.t_duration_sec for s in stats.values()]) <= t_elapsed_max
