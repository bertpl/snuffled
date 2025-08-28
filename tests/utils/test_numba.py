import math

import numpy as np
import pytest

from snuffled._core.utils.numba import clip_scalar, geomean


@pytest.mark.parametrize(
    "a, a_min, a_max, expected_result",
    [
        (0.0, -1.0, 1.0, 0.0),
        (-5.0, -1.0, 1.0, -1.0),
        (17.1, -1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
    ],
)
def test_numba_clip_scalar(a: float, a_min: float, a_max: float, expected_result: float):
    # --- act & assert ------------------------------------
    assert clip_scalar(a, a_min, a_max) == expected_result


@pytest.mark.parametrize(
    "values, expected_result, tol",
    [
        ([], 1.0, 0.0),
        ([math.pi], math.pi, 0.0),
        ([1, 2, 3, 4, 0], 0.0, 0.0),
        ([1, 2, 4, 8, 16], 4.0, 1e-15),
        ([3, 9, 27], 9.0, 1e-15),
        ([4, 17], math.sqrt(4 * 17), 1e-15),
        ([1e-100] * 100, 1e-100, 1e-12),  # naive implementations will experience underflow here
        ([1e100] * 100, 1e100, 1e-12),  # naive implementations will experience overflow here
    ],
)
def test_numba_geomean(values: list[float], expected_result: float, tol: float):
    # --- arrange -----------------------------------------
    values_arr = np.array(values, dtype=np.float64)

    # --- act ---------------------------------------------
    result = geomean(values_arr)

    # --- assert ------------------------------------------
    assert result == pytest.approx(expected_result, rel=tol, abs=tol)


@pytest.mark.parametrize(
    "values",
    [
        [-1.0, 2.0, 4.0],
        [-1.0, -2.0, 4.0, 10.0],
    ],
)
def test_numba_geomean_value_error(values: list[float]):
    # --- arrange -----------------------------------------
    values_arr = np.array(values, dtype=np.float64)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = geomean(values_arr)
