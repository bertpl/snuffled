import pytest

from snuffled._core.utils.signs import robust_sign_estimate


@pytest.mark.parametrize(
    "values, expected_result",
    [
        ([1, 2, 3], 1),
        ([0, 1, 2], 1),
        ([0, 0, 0], 0),
        ([0, -1, -2], -1),
        ([-1, -2, -3], -1),
        ([-100, 2, 3, 4, 5], 1),
        ([-10, 2, 3, 4, 5], 1),
        ([-1, 2, 3, 4, 5], 1),
        ([-100, -3, 3, 4, 5], 1),
        ([-10, -3, 3, 4, 5], 1),
        ([-1, -1, 3, 4, 5], 1),
        ([-100, -4, 3, 4, 5], 0),
        ([-4, -4, 3, 4, 5], 0),
        ([-4, -4, 3, 4, 100], 0),
    ],
)
def test_robust_sign_estimate(values: list[float], expected_result: int):
    # --- act ---------------------------------------------
    result = robust_sign_estimate(values)

    # --- assert ------------------------------------------
    assert result == expected_result
