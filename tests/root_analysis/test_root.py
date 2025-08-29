from contextlib import nullcontext

import pytest

from snuffled._core.models.root_analysis import Root


@pytest.mark.parametrize(
    "x_min, x_max, deriv_sign, expectation_mgr",
    [
        (-1.0, 2.0, -1, nullcontext()),
        (1.0, 1.1, +1, nullcontext()),
        (-1.0, 2.0, -1.0, nullcontext()),
        (1.0, 1.1, +1.0, nullcontext()),
        (1.1, 1.1, +1.0, nullcontext()),
        (1.2, 1.1, +1.0, pytest.raises(ValueError)),
        (1.0, 1.1, 0.0, pytest.raises(ValueError)),
        (1.0, 1.1, 2.0, pytest.raises(ValueError)),
    ],
)
def test_root_validation(x_min: float, x_max: float, deriv_sign: int, expectation_mgr):
    # --- act & assert ------------------------------------
    with expectation_mgr:
        _ = Root(x_min=x_min, x_max=x_max, deriv_sign=deriv_sign)


def test_root_width():
    # --- arrange -----------------------------------------
    r = Root(x_min=1.1235, x_max=5.23234, deriv_sign=1)

    # --- act ---------------------------------------------
    w = r.width

    # --- assert ------------------------------------------
    assert w == (5.23234 - 1.1235)
