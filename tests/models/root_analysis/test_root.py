from contextlib import nullcontext

import pytest

from snuffled._core.models.root_analysis import Root


@pytest.mark.parametrize(
    "x_min, x, x_max, fx_min, fx, fx_max, expectation_mgr",
    [
        (-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, nullcontext()),
        (-1.0, -1.0, 1.0, -2.0, -2.0, 2.0, nullcontext()),
        (-1.0, 1.0, 1.0, -2.0, 2.0, 2.0, nullcontext()),
        (-1.0, 0.0, 1.0, 2.0, 0.0, 2.0, nullcontext()),
        (-1.0, -1.0, 1.0, -2.0, -1.9, 2.0, pytest.raises(ValueError)),
        (-1.0, 1.0, 1.0, -2.0, 1.9, 2.0, pytest.raises(ValueError)),
        (-2.0, -2.1, 3.0, -4.0, -4.2, 6.0, pytest.raises(ValueError)),
        (-2.0, 3.1, 3.0, -4.0, 6.2, 6.0, pytest.raises(ValueError)),
        (-1.0, 0.0, 1.0, 2.0, 0.1, 2.0, pytest.raises(ValueError)),
        (-1.0, 0.0, 1.0, 2.0, -0.1, 2.0, pytest.raises(ValueError)),
    ],
    ids=[
        "ok (x_min<x<x_max)",
        "ok (x=x_min)",
        "ok (x=x_max)",
        "ok (even root)",
        "nok (fx!=fx_min)",
        "nok (fx!=fx_max)",
        "nok (x<x_min)",
        "nok (x>x_max)",
        "nok (even root 1)",
        "nok (even root 2)",
    ],
)
def test_root_validation(
    x_min: float, x: float, x_max: float, fx_min: float, fx: float, fx_max: float, expectation_mgr
):
    # --- act & assert ------------------------------------
    with expectation_mgr:
        _ = Root(
            x_min=x_min,
            x=x,
            x_max=x_max,
            fx_min=fx_min,
            fx=fx,
            fx_max=fx_max,
        )


def test_root_width():
    # --- arrange -----------------------------------------
    r = Root(x_min=1.1235, x=2.0, x_max=5.23234, fx_min=-1.0, fx=0.0, fx_max=1.0)

    # --- act ---------------------------------------------
    w = r.width

    # --- assert ------------------------------------------
    assert w == (5.23234 - 1.1235)


@pytest.mark.parametrize(
    "fx_min, fx_max, expected_result",
    [
        (-1.23, +2.34, +1),
        (+1.23, -2.34, -1),
        (-1.23, -2.34, 0),
        (+1.23, +2.34, 0),
    ],
)
def test_root_deriv_sign(fx_min: float, fx_max: float, expected_result: int):
    # --- arrange -----------------------------------------
    r = Root(x_min=-1.0, x=0.0, x_max=1.0, fx_min=fx_min, fx=0.0, fx_max=fx_max)

    # --- act ---------------------------------------------
    result = r.deriv_sign

    # --- assert ------------------------------------------
    assert result == expected_result
