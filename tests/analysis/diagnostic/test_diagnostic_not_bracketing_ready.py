import math
from functools import partial
from typing import Callable

import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.diagnostic import DiagnosticAnalyser
from snuffled._core.models import Diagnostic


# =================================================================================================
#  Test functions
# =================================================================================================
def f_linear(x: float, fx_left: float, fx_right: float) -> float:
    """Linear function going through (-1, fx_left) and (1, fx_right)"""
    return ((x - 1) / 2) * fx_left + ((x + 1) / 2) * fx_right


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize(
    "fun, expected_result",
    [
        (partial(f_linear, fx_left=-1, fx_right=1), 0.0),
        (partial(f_linear, fx_left=-1e-9, fx_right=1), 0.0),
        (partial(f_linear, fx_left=-1, fx_right=1e-9), 0.0),
        (partial(f_linear, fx_left=1e-250, fx_right=-1e-250), 0.0),
        (partial(f_linear, fx_left=0, fx_right=1), 0.5),
        (partial(f_linear, fx_left=-1, fx_right=0), 0.5),
        (partial(f_linear, fx_left=-1e-250, fx_right=0), 0.5),
        (partial(f_linear, fx_left=1, fx_right=2), 1.0),
        (partial(f_linear, fx_left=-1, fx_right=-2), 1.0),
        (partial(f_linear, fx_left=1e-250, fx_right=1e-250), 1.0),
    ],
)
def test_function_analyser_not_bracketing_ready(fun: Callable[[float], float], expected_result: float):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-9, rel_tol_scale=10.0)
    analyser = DiagnosticAnalyser(sampler)

    # --- act ---------------------------------------------
    result = analyser.extract(Diagnostic.INTERVAL_NOT_BRACKETING_READY)

    # --- assert ------------------------------------------
    assert result == expected_result
