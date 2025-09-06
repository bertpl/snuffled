import math

import numpy as np
import pytest

from snuffled._core.analysis.roots.curve_fitting import compute_x_deltas
from snuffled._core.analysis.roots.single_root_one_side_analyser import SingleRootOneSideAnalyser
from snuffled._core.utils.noise import noise_from_float


# =================================================================================================
#  Pre-Processing
# =================================================================================================
@pytest.mark.parametrize(
    "dx, x_deltas, fx_values, dx_sign, fx_sign",
    [
        (0.1, [0.1, 0.2, 0.4], [1.0, 2.0, 4.1], +1, +1),
        (0.1, [0.1, 0.2, 0.4], [-1.0, -2.0, -4.1], -1, -1),
        (0.01, [0.01, 0.02, 0.04], [1.0, -1.0, 4.5], +1, +1),
        (0.1, [0.1, 0.2, 0.4], [0.0, 0.0, 0.0], +1, +1),
        (0.1, [0.1, 0.2, 0.4], [0.0, -0.1, -0.1], +1, +1),
    ],
    ids=[
        "regular",
        "negative_dx_sign_fx_sign",
        "one_negative_fx",
        "all_zero_fx",
        "no_positive_fx",
    ],
)
def test_single_root_one_side_analyser_preprocess_x_fx(
    dx: float,
    x_deltas: list[float],
    fx_values: list[float],
    dx_sign: int,
    fx_sign: int,
):
    # --- arrange -----------------------------------------
    x_deltas_arr = np.array(x_deltas)
    fx_values_arr = np.array(fx_values)

    # --- act ---------------------------------------------
    analyser = SingleRootOneSideAnalyser(dx, x_deltas_arr, fx_values_arr, dx_sign, fx_sign)

    # --- assert ------------------------------------------

    # check if core data is stored correctly
    assert analyser.dx == dx
    assert np.array_equal(analyser.x_deltas, x_deltas)
    assert np.array_equal(analyser.fx_values, fx_values)
    assert analyser.fx_sign == fx_sign
    assert analyser.dx_sign == dx_sign

    # check if pre-processing invariants are respected
    assert analyser.x_scale > 0
    assert analyser.fx_scale > 0
    assert np.median(analyser.x_pre) == pytest.approx(1.0, rel=1e-15, abs=1e-15)
    assert np.allclose(analyser.x_scale * analyser.x_pre, x_deltas_arr, rtol=1e-15)
    assert np.allclose(analyser.fx_sign * analyser.fx_scale * analyser.fx_pre, fx_values_arr, rtol=1e-15)


# =================================================================================================
#  Parameter extraction
# =================================================================================================
@pytest.mark.parametrize("fx_sign", [+1, -1])
@pytest.mark.parametrize("c_true", [1.0, -1.0, 0.456, 1.893, -2.235, -0.345])
@pytest.mark.parametrize("c_noise", [0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1])
def test_single_root_one_side_analyser_abc_params(fx_sign: int, c_true: float, c_noise: float):
    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        return fx_sign * (_x**c_true) * (1 + (c_noise * noise_from_float(_x)))

    dx = 1e-10
    dx_sign = +1
    x_deltas = compute_x_deltas(dx=dx, k=10, seed=42)
    fx_values = np.array([fun(x) for x in x_deltas])

    # --- act ---------------------------------------------
    analyser = SingleRootOneSideAnalyser(dx, x_deltas, fx_values, dx_sign, fx_sign)
    a_min, a_max = analyser._a_min_max
    b_min, b_max = analyser._b_min_max
    c_min, c_max = analyser._c_min_max

    # --- assert ------------------------------------------
    assert a_min <= 1.0 <= a_max  # additional check if scaling works well for these simple cases
    assert b_min <= 0.0 <= b_max  # we didn't add a constant term in fun(x)
    assert c_min <= c_true <= c_max


@pytest.mark.parametrize("fx_sign", [+1, -1])
@pytest.mark.parametrize("c_true", [1.0, -1.0, 0.456, 1.893, -2.235, -0.345])
@pytest.mark.parametrize("c_noise", [0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1])
def test_single_root_one_side_analyser_fx_values(fx_sign: int, c_true: float, c_noise: float):
    # --- arrange -----------------------------------------
    def fun_true(_x: float) -> float:
        return fx_sign * (_x**c_true)

    def fun_with_noise(_x: float) -> float:
        return fun_true(_x) * (1 + (c_noise * noise_from_float(_x)))

    dx = 1e-10
    dx_sign = +1
    x_deltas = compute_x_deltas(dx=dx, k=10, seed=42)
    fx_values = np.array([fun_with_noise(x) for x in x_deltas])

    fx_1_true = fun_true(dx)
    fx_2_true = fun_true(2 * dx)
    fx_4_true = fun_true(4 * dx)

    # --- act ---------------------------------------------
    analyser = SingleRootOneSideAnalyser(dx, x_deltas, fx_values, dx_sign, fx_sign)
    fx_1_min, fx_1_max = analyser.f1
    fx_2_min, fx_2_max = analyser.f2
    fx_4_min, fx_4_max = analyser.f4

    # --- assert ------------------------------------------
    print(fx_1_min, fx_1_true, fx_1_max)
    print(fx_2_min, fx_2_true, fx_2_max)
    print(fx_4_min, fx_4_true, fx_4_max)

    assert fx_1_min <= fx_1_true <= fx_1_max
    assert fx_2_min <= fx_2_true <= fx_2_max
    assert fx_4_min <= fx_4_true <= fx_4_max


# =================================================================================================
#  Property extraction
# =================================================================================================
@pytest.mark.parametrize(
    "c_noise, min_value, max_value",
    [
        (0.0, 0.00, 0.00),
        (1e-10, 0.10, 0.90),
        (1e-6, 1.00, 1.00),
        (0.1, 1.00, 1.00),
        (1.0, 1.00, 1.00),
    ],
)
def test_single_root_one_side_analyser_ill_behaved(c_noise: float, min_value: float, max_value: float):
    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        return _x + (c_noise * noise_from_float(_x))

    dx = 1e-10
    dx_sign = +1
    fx_sign = +1
    x_deltas = compute_x_deltas(dx=dx, k=10, seed=42)
    fx_values = np.array([fun(x) for x in x_deltas])

    # --- act ---------------------------------------------
    analyser = SingleRootOneSideAnalyser(dx, x_deltas, fx_values, dx_sign, fx_sign)
    value = analyser.ill_behaved

    # --- assert ------------------------------------------
    assert min_value <= value <= max_value


@pytest.mark.parametrize(
    "c_true, c_noise, min_value, max_value",
    [
        (-2.0, 0.0, 0.00, 0.00),
        (-1.0, 0.0, 0.00, 0.00),
        (-0.5, 0.0, 0.00, 0.00),
        (0.9, 0.0, 0.00, 0.00),
        (1.0, 0.0, 0.00, 0.01),
        (math.sqrt(2), 0.0, 0.45, 0.55),
        (2.0, 0.0, 0.99, 1.00),
        (2.1, 0.0, 1.00, 1.00),
        (0.9, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (1.0, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (math.sqrt(2), 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (2.0, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (2.1, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
    ],
)
def test_single_root_one_side_analyser_deriv_zero(c_true: float, c_noise: float, min_value: float, max_value: float):
    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        return (_x**c_true) + (c_noise * noise_from_float(_x))

    dx = 1e-10
    dx_sign = +1
    fx_sign = +1
    x_deltas = compute_x_deltas(dx=dx, k=10, seed=42)
    fx_values = np.array([fun(x) for x in x_deltas])

    # --- act ---------------------------------------------
    analyser = SingleRootOneSideAnalyser(dx, x_deltas, fx_values, dx_sign, fx_sign)
    value = analyser.deriv_zero

    # --- assert ------------------------------------------
    assert min_value <= value <= max_value


@pytest.mark.parametrize(
    "c_true, c_noise, min_value, max_value",
    [
        (-2.0, 0.0, 1.00, 1.00),
        (-1.0, 0.0, 1.00, 1.00),
        (-0.5, 0.0, 1.00, 1.00),
        (0.4, 0.0, 1.00, 1.00),
        (0.5, 0.0, 0.95, 1.00),  # slightly looser bounds than deriv_zero test - this case is more ill-conditioned
        (1 / math.sqrt(2), 0.0, 0.45, 0.55),
        (1.0, 0.0, 0.00, 0.01),
        (1.1, 0.0, 0.00, 0.00),
        (0.4, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (0.5, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (1 / math.sqrt(2), 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (1.0, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
        (1.1, 1.0, 0.00, 0.01),  # benefit of the doubt in case of very noisy function
    ],
)
def test_single_root_one_side_analyser_deriv_infinite(
    c_true: float, c_noise: float, min_value: float, max_value: float
):
    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        return (_x**c_true) + (c_noise * noise_from_float(_x))

    dx = 1e-10
    dx_sign = +1
    fx_sign = +1
    x_deltas = compute_x_deltas(dx=dx, k=10, seed=42)
    fx_values = np.array([fun(x) for x in x_deltas])

    # --- act ---------------------------------------------
    analyser = SingleRootOneSideAnalyser(dx, x_deltas, fx_values, dx_sign, fx_sign)
    value = analyser.deriv_infinite

    # --- assert ------------------------------------------
    assert min_value <= value <= max_value


@pytest.mark.parametrize(
    "c_true, offset, c_noise, min_value, max_value",
    [
        (1.0, 0.0, 0.0, 0.00, 0.00),
        (1.0, 2e-10, 0.0, 0.45, 0.55),  # offset 50% of f(2*dx)
        (1.0, 1.0, 0.0, 0.99, 1.00),
        (1.0, 1.0, 0.01, 0.10, 0.90),  # reduced certainty in presence of some noise
        (1.0, 0.0, 1.0, 0.00, 0.00),  # benefit of the doubt in case of very noisy function
        (1.0, 2e-10, 1.0, 0.00, 0.00),  # benefit of the doubt in case of very noisy function
        (1.0, 1.0, 1.0, 0.00, 0.00),  # benefit of the doubt in case of very noisy function
        (-1.0, 0.0, 0.0, 1.00, 1.00),
        (0.01, 0.0, 0.0, 0.80, 0.95),  # at some point, a very high-power root will start looking like a step
        (0.001, 0.0, 0.0, 0.95, 0.99),  # at some point, a very high-power root will start looking like a step
        (0.0001, 0.0, 0.0, 0.99, 1.00),  # at some point, a very high-power root will start looking like a step
    ],
)
def test_single_root_one_side_analyser_discontinuous(
    c_true: float, offset: float, c_noise: float, min_value: float, max_value: float
):
    # --- arrange -----------------------------------------
    def fun(_x: float) -> float:
        return offset + (_x**c_true) + (c_noise * noise_from_float(_x))

    dx = 1e-10
    dx_sign = +1
    fx_sign = +1
    x_deltas = compute_x_deltas(dx=dx, k=10, seed=42)
    fx_values = np.array([fun(x) for x in x_deltas])

    # --- act ---------------------------------------------
    analyser = SingleRootOneSideAnalyser(dx, x_deltas, fx_values, dx_sign, fx_sign)
    value = analyser.discontinuous

    # --- assert ------------------------------------------
    assert min_value <= value <= max_value
