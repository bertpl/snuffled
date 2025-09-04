import numpy as np
import pytest

from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.roots.single_root_two_side_analyser import SingleRootTwoSideAnalyser
from snuffled._core.models import RootProperty
from snuffled._core.utils.noise import noise_from_float


# =================================================================================================
#  Properties
# =================================================================================================
@pytest.mark.parametrize(
    "c_true, c_noise, offset_neg, offset_pos, expectations",
    [
        (
            3.0,
            0.0,
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
            0.0,
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
            0.4,
            0.0,
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
            0.4,
            10.0,
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.99, 1.00),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            3.0,
            10.0,
            0.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.00, 0.00),
                RootProperty.ILL_BEHAVED: (0.99, 1.00),
                RootProperty.ASYMMETRIC: (0.00, 0.01),
            },
        ),
        (
            1.0,
            0.0,
            1.0,
            0.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.49, 0.51),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.99, 1.00),
            },
        ),
        (
            1.0,
            0.0,
            0.0,
            1.0,
            {
                RootProperty.DERIVATIVE_ZERO: (0.00, 0.00),
                RootProperty.DERIVATIVE_INFINITE: (0.00, 0.00),
                RootProperty.DISCONTINUOUS: (0.49, 0.51),
                RootProperty.ILL_BEHAVED: (0.00, 0.01),
                RootProperty.ASYMMETRIC: (0.99, 1.00),
            },
        ),
        (
            1.0,
            0.0,
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
    ],
)
def test_single_root_two_side_analyser_all_props(
    c_true: float,
    c_noise: float,
    offset_neg: float,
    offset_pos: float,
    expectations: dict[RootProperty, tuple[float, float]],
):
    """Test properties that are simply an average of left- and right-side root analysis"""

    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        _fx = np.sign(_x) * (abs(_x) ** c_true) * (1 + c_noise * noise_from_float(_x))
        if _x > 0:
            _fx += offset_pos
        elif _x < 0:
            _fx -= offset_neg
        return _fx

    dx = 1e-10
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, dx=dx, seed=42)
    root = sampler.roots()[0]

    # --- act ---------------------------------------------
    analyser = SingleRootTwoSideAnalyser(sampler, root, n_root_samples=100, seed=1234)

    # --- assert ------------------------------------------
    for root_prop, (min_value, max_value) in expectations.items():
        assert min_value <= analyser.extract(root_prop) <= max_value, f"{root_prop} not in [{min_value},{max_value}]"


@pytest.mark.parametrize(
    "c_neg, c_pos, min_value, max_value",
    [
        (1.0, 1.0, 0.00, 0.00),
        (1.0, 2.0, 0.49, 0.51),
        (0.5, 1.0, 0.49, 0.51),
        (1.0, 10.0, 0.89, 0.91),
        (1e-3, 1e3, 0.99, 1.00),
    ],
)
def test_single_root_two_side_analyser_asymmetric_linear(
    c_neg: float, c_pos: float, min_value: float, max_value: float
):
    """Test asymmetry property - by changing slopes of 2 linear functions"""

    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        if _x >= 0.0:
            return c_pos * _x
        else:
            return c_neg * _x

    dx = 1e-10
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, dx=dx, seed=42)
    root = sampler.roots()[0]

    # --- act ---------------------------------------------
    analyser = SingleRootTwoSideAnalyser(sampler, root, n_root_samples=100, seed=1234)
    value = analyser.extract(RootProperty.ASYMMETRIC)

    # --- assert ------------------------------------------
    assert min_value <= value <= max_value, f"RootProperty.ASYMMETRIC not in [{min_value},{max_value}]"


@pytest.mark.parametrize(
    "e_neg, e_pos, min_value, max_value",
    [
        (1.0, 1.0, 0.00, 0.00),
        (2.0, 2.0, 0.00, 0.00),
        (0.999, 1.001, 0.01, 0.10),
        (0.99, 1.01, 0.10, 0.50),
        (0.95, 1.05, 0.50, 0.95),
        (0.9, 1.1, 0.95, 1.00),
        (0.5, 2.0, 0.99, 1.00),
    ],
)
def test_single_root_two_side_analyser_asymmetric_nonlinear(
    e_neg: float, e_pos: float, min_value: float, max_value: float
):
    """Test asymmetry property - by changing exponents of 2 functions"""

    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        if _x >= 0.0:
            return _x**e_pos
        else:
            return -(abs(_x) ** e_neg)

    dx = 1e-10
    sampler = FunctionSampler(fun=fun, x_min=-1.0, x_max=1.0, dx=dx, seed=42)
    root = sampler.roots()[0]

    # --- act ---------------------------------------------
    analyser = SingleRootTwoSideAnalyser(sampler, root, n_root_samples=100, seed=1234)
    value = analyser.extract(RootProperty.ASYMMETRIC)

    # --- assert ------------------------------------------
    assert min_value <= value <= max_value, f"RootProperty.ASYMMETRIC not in [{min_value},{max_value}]"
