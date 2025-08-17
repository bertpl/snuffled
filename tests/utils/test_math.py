import numpy as np
import pytest

from snuffled._core.utils.math import smooth_sign, smooth_sign_array


@pytest.mark.parametrize(
    "x, inner_tol, outer_tol, min_result, max_result",
    [
        (0.0, 0.1, 0.2, 0.0, 0.0),
        (0.0, 0.0, 0.2, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 1.0, 1.0),
        (0.1, 0.1, 0.4, 0.25, 0.25),
        (0.2, 0.1, 0.4, 0.40, 0.60),
        (0.4, 0.1, 0.4, 0.75, 0.75),
        (0.1, 0.2, 0.2, 0.25, 0.25),
        (0.2, 0.2, 0.2, 0.40, 0.60),
        (0.4, 0.2, 0.2, 0.75, 0.75),
        (0.1, 0.0, 0.4, 0.25, 0.25),
        (0.2, 0.0, 0.4, 0.40, 0.60),
        (0.4, 0.0, 0.4, 0.75, 0.75),
        (100, 0.1, 0.4, 1.00, 1.00),
        (0.02, 0.2, 2.5, 0.00, 0.05),
        (0.2, 0.2, 2.5, 0.25, 0.25),
        (2.5, 0.2, 2.5, 0.75, 0.75),
        (1e3, 0.1, 0.4, 1.00, 1.00),
    ],
)
def test_smooth_sign(x: float, inner_tol: float, outer_tol: float, min_result: float, max_result: float):
    # --- act ---------------------------------------------
    result_pos_x = smooth_sign(x, inner_tol, outer_tol)
    result_neg_x = smooth_sign(-x, inner_tol, outer_tol)
    # --- assert ------------------------------------------
    assert min_result * (1 - 1e-14) <= result_pos_x <= max_result * (1 + 1e-14)
    assert result_neg_x == -result_pos_x


@pytest.mark.parametrize(
    "x, inner_tol, outer_tol, expected_result",
    [
        (
            np.array([0.0, 0.1, 0.4, 100.0]),
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.4, 0.4, 0.4, 0.4]),
            np.array([0.0, 0.25, 0.75, 1.0]),
        ),
    ],
)
def test_smooth_sign_array(x: np.ndarray, inner_tol: np.ndarray, outer_tol: np.ndarray, expected_result: np.ndarray):
    # --- act ---------------------------------------------
    result = smooth_sign_array(x, inner_tol, outer_tol)

    # --- assert ------------------------------------------
    assert np.allclose(result, expected_result, rtol=1e-14)
