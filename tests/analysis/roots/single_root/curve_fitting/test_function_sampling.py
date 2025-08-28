import math

import numpy as np
import pytest

from snuffled._core.analysis.roots.single_root.curve_fitting import compute_x_deltas


@pytest.mark.parametrize(
    "dx, k",
    [
        (1e-3, 1),
        (1e-3, 2),
        (1e-6, 3),
        (1e-9, 5),
        (1e-9, 1000),
    ],
)
def test_compute_x_deltas(dx: float, k: int):
    # --- arrange -----------------------------------------
    len_expected = 3 + 6 * k
    seed1 = 1234
    seed2 = 2345

    def geomean(_v: np.ndarray) -> float:
        return 2 ** float(np.mean(np.log2(_v)))

    # --- act ---------------------------------------------
    x_deltas = compute_x_deltas(dx, k, seed=seed1)
    x_deltas_2 = compute_x_deltas(dx, k, seed=seed1)
    x_deltas_3 = compute_x_deltas(dx, k, seed=seed2)

    # --- assert ------------------------------------------

    # check seed handling
    assert np.array_equal(x_deltas, x_deltas_2)
    assert not np.array_equal(x_deltas, x_deltas_3)

    # check counts and sorting
    assert len(x_deltas) == len_expected, "incorrect number of values"
    assert len(x_deltas) == len(set(x_deltas)), "values should be unique"
    assert np.array_equal(np.sort(x_deltas), x_deltas), "values should be sorted"

    # check statistical properties - overall
    assert dx / math.sqrt(2) <= min(x_deltas) <= dx
    assert 4 * dx <= max(x_deltas) <= 4 * dx * math.sqrt(2)
    assert np.median(x_deltas) == 2 * dx
    assert geomean(x_deltas) == pytest.approx(2 * dx, rel=1e-15)
    assert min(x_deltas) == pytest.approx(2.0**-0.5 * dx, rel=1e-15)
    assert max(x_deltas) == pytest.approx(2.0**2.5 * dx, rel=1e-15)

    # check statistical properties - per group
    group_1 = x_deltas[: len_expected // 3]
    group_2 = x_deltas[len_expected // 3 : 2 * len_expected // 3]
    group_3 = x_deltas[2 * len_expected // 3 :]

    assert dx / math.sqrt(2) <= min(group_1) <= dx
    assert 2 * dx / math.sqrt(2) <= min(group_2) <= 2 * dx
    assert 4 * dx / math.sqrt(2) <= min(group_3) <= 4 * dx

    assert dx <= max(group_1) <= dx * math.sqrt(2)
    assert 2 * dx <= max(group_2) <= 2 * dx * math.sqrt(2)
    assert 4 * dx <= max(group_3) <= 4 * dx * math.sqrt(2)

    assert np.median(group_1) == dx
    assert np.median(group_2) == 2 * dx
    assert np.median(group_3) == 4 * dx

    assert geomean(group_1) == pytest.approx(dx, rel=1e-15)
    assert geomean(group_2) == pytest.approx(2 * dx, rel=1e-15)
    assert geomean(group_3) == pytest.approx(4 * dx, rel=1e-15)
