import numpy as np
import pytest

from snuffled._core.analysis._function_sampler import smoothen_fx_abs_tol, smoothen_fx_rel_tol


@pytest.mark.parametrize(
    "fx_values, abs_tol, expected_result",
    [
        (
            [0.1, 0.2, 0.3],
            1e-3,
            [0.1, 0.2, 0.3],
        ),
        (
            [0.1, 0.2, 0.3],
            1e-2,
            [0.1, 0.2, 0.3],
        ),
        (
            [0.1, 0.2, 0.3],
            0.15,
            [0.1, 0.1, 0.3],
        ),
        (
            [0.1, 0.2, 0.3],
            1.0,
            [0.1, 0.1, 0.1],
        ),
        (
            [0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.0, 1.1],
            0.15,
            [0.1, 0.1, 0.3, 0.3, 0.8, 0.8, 1.0, 1.0],
        ),
    ],
)
def test_smoothen_fx_abs_tol(fx_values: list[float], abs_tol: float, expected_result: list[float]):
    # --- act ---------------------------------------------
    smoothed = smoothen_fx_abs_tol(fx=np.array(fx_values), abs_tol=abs_tol)

    # --- assert ------------------------------------------
    assert np.array_equal(expected_result, smoothed)


@pytest.mark.parametrize(
    "fx_values, rel_tol, expected_result",
    [
        (
            [1.000, 1.001, 1.002, 1.003, 1.004, 2.000, 2.001, 2.002, 2.003],
            1e-3,
            [1.000, 1.000, 1.002, 1.002, 1.004, 2.000, 2.000, 2.000, 2.003],
        ),
        (
            [1.000, 1.001, 1.002, 1.003, 1.004, 2.000, 2.001, 2.002, 2.003, 2.005],
            2e-3,
            [1.000, 1.000, 1.000, 1.003, 1.003, 2.000, 2.000, 2.000, 2.000, 2.005],
        ),
        (
            [1.000, 1.001, 1.002, 1.003, 1.004, 2.000, 2.001, 2.002, 2.003],
            1e-6,
            [1.000, 1.001, 1.002, 1.003, 1.004, 2.000, 2.001, 2.002, 2.003],
        ),
        (
            [1.000, 1.001, 1.002, 1.003, 1.004, 2.000, 2.001, 2.002, 2.003],
            1e-1,
            [1.000, 1.000, 1.000, 1.000, 1.000, 2.000, 2.000, 2.000, 2.000],
        ),
    ],
)
def test_smoothen_fx_rel_tol(fx_values: list[float], rel_tol: float, expected_result: list[float]):
    # --- act ---------------------------------------------
    smoothed = smoothen_fx_rel_tol(fx=np.array(fx_values), rel_tol=rel_tol)

    # --- assert ------------------------------------------
    assert np.array_equal(expected_result, smoothed)
