import math
from typing import Callable

import numpy as np
import pytest

from snuffled._core.analysis.function.helpers_discontinuous import compute_discontinuity_score


# =================================================================================================
#  Helpers
# =================================================================================================
def f_score_linear(x: float) -> float:
    return x


def f_score_quad(x: float) -> float:
    return x + (x * x) / 10


def f_score_exp(x: float) -> float:
    return math.exp(10 * x) - 0.5


def f_score_sign(x: float) -> float:
    return float(np.sign(x))


@pytest.mark.parametrize(
    "fun, x_values, dx_min, min_score, max_score",
    [
        (f_score_linear, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-9),
        (f_score_linear, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.0, 1e-9),
        (f_score_quad, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-6),
        (f_score_quad, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.0, 1e-6),
        (f_score_exp, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-3),
        (f_score_exp, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.0, 1e-3),
        (f_score_sign, [-1.0, -0.2, 0.2, 1.0], 1e-10, 0.0, 1e-3),
        (f_score_sign, [-1.0, -1e-9, 0.0, 1e-9, 1.0], 1e-10, 0.999, 1.0),
    ],
)
def test_compute_discontinuity_score(
    fun: Callable[[float], float],
    x_values: list[float],
    dx_min: float,
    min_score: float,
    max_score: float,
):
    # --- arrange -----------------------------------------
    x_values = np.array(x_values)
    fx_values = np.array([fun(x) for x in x_values])

    # --- act ---------------------------------------------
    score = compute_discontinuity_score(x_values, fx_values, dx_min)

    print(score)

    # --- assert ------------------------------------------
    assert min_score <= score <= max_score
