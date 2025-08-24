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
def f_linear(x: float) -> float:
    return x - 1 / math.pi


def f_quadratic(x: float, c0: float) -> float:
    # no zeroes for c0>0 or c0<-1
    return c0 + (x * x)


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize(
    "fun, expected_result",
    [
        (f_linear, 0.0),
        (partial(f_quadratic, c0=1), 1.0),
        (partial(f_quadratic, c0=0.01), 1.0),
        (partial(f_quadratic, c0=0), 1.0),
        (partial(f_quadratic, c0=-0.01), 0.0),
        (partial(f_quadratic, c0=-0.1), 0.0),
        (partial(f_quadratic, c0=-0.5), 0.0),
        (partial(f_quadratic, c0=-0.9), 0.0),
        (partial(f_quadratic, c0=-0.99), 0.0),
        (partial(f_quadratic, c0=-0.99), 0.0),
        (partial(f_quadratic, c0=-1.0), 0.0),
        (partial(f_quadratic, c0=-1.001), 1.0),
        (partial(f_quadratic, c0=-2), 1.0),
    ],
)
def test_function_analyser_no_zeros_detected(fun: Callable[[float], float], expected_result: float):
    # --- arrange -----------------------------------------
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, n_fun_samples=1000, dx=1e-9, rel_tol_scale=10.0)
    analyser = DiagnosticAnalyser(sampler)

    # --- act ---------------------------------------------
    result = analyser.extract(Diagnostic.NO_ZEROS_DETECTED)

    # --- assert ------------------------------------------
    assert result == expected_result
