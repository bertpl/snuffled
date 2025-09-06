import math

import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.roots.roots_analyser import RootsAnalyser
from snuffled._core.models import RootProperty


# =================================================================================================
#  Regular cases
# =================================================================================================
@pytest.mark.parametrize(
    "c_neg, c_pos, e_neg, e_pos, offset_neg, offset_pos, expectations",
    [
        (
            1.0,
            1.0,
            3.0,
            3.0,
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (1.00, 1.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            1.0,
            1.0,
            math.sqrt(2),
            math.sqrt(2),
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.45, 0.55),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            1.0,
            1.0,
            1 / math.sqrt(2),
            1 / math.sqrt(2),
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.45, 0.55),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            1.0,
            1.0,
            1 / 3,
            1 / 3,
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (1.00, 1.00),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            0.5,
            2.0,
            1.0,
            1.0,
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.70, 0.80),
            },
        ),
        (
            1.0,
            1.0,
            1 / 3,
            3.0,
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.49, 0.51),
                RootProperty.DERIVATIVE_INFINITE: (0.49, 0.51),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.99, 1.00),
            },
        ),
        (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.49, 0.50),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.99, 1.00),
            },
        ),
        (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.99, 1.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            10.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.99, 1.00),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.89, 0.91),
            },
        ),
    ],
)
@pytest.mark.parametrize("base_function", ["sin", "lin_quad", "exp"])
def test_roots_analyser_properties(
    base_function: str,
    c_neg: float,
    c_pos: float,
    e_neg: float,
    e_pos: float,
    offset_neg: float,
    offset_pos: float,
    expectations: dict[RootProperty, tuple[float, float]],
):
    """Test properties that are simply an average of left- and right-side root analysis"""

    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        if base_function == "sin":
            _tmp = math.sin(10 * _x + 0.01)
        elif base_function == "lin_quad":
            _tmp = (_x - 0.1) + 0.1 * _x * _x
        else:
            _tmp = math.exp(2 * _x) - 0.5

        if _tmp >= 0:
            _fx = c_pos * (_tmp**e_pos)
        else:
            _fx = -c_neg * (abs(_tmp) ** e_neg)

        if _fx > 0:
            _fx += offset_pos
        elif _fx < 0:
            _fx -= offset_neg
        return _fx

    dx = 1e-10
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, dx=dx, seed=42)

    # --- act ---------------------------------------------
    analyser = RootsAnalyser(sampler, n_root_samples=100, seed=42)

    print(analyser.extract_all().as_dict())

    # --- assert ------------------------------------------
    for root_prop, (min_value, max_value) in expectations.items():
        assert min_value <= analyser.extract(root_prop) <= max_value, f"{root_prop} not in [{min_value},{max_value}]"


# =================================================================================================
#  Edge cases
# =================================================================================================
def test_roots_analyser_no_roots():
    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        return 1.0 + 0.1 * _x  # no roots in [-1,1]

    dx = 1e-10
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, dx=dx, seed=42)

    # --- act ---------------------------------------------
    analyser = RootsAnalyser(sampler, n_root_samples=100, seed=42)
    all_props = analyser.extract_all()

    # --- assert ------------------------------------------
    for value in all_props.as_array():
        assert value == 0.0
