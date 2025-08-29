import numpy as np

from snuffled._core.compatibility import numba


@numba.njit(inline="always")
def fitting_cost(x: np.ndarray, fx: np.ndarray, a: float, b: float, c: float, reg: float) -> float:
    """
    L1-cost of fitting fitting_curve to (x,fx)-values, where all x>0.
    A regularization term is added that should help keep c=1.0 and b=0.0 unless the data convincingly says
    otherwise.
    """
    fx_pred = fitting_curve(x, a, b, c)
    reg_term = reg * (abs(b) + abs(np.log10(c)))
    return float(np.mean(np.abs(fx - fx_pred)) + reg_term)


@numba.njit(inline="always")
def fitting_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """curve g(x) = a * (b + (1-b)**(x^c), assuming x>0"""
    return a * (b + (1 - b) * (np.pow(x, c)))


@numba.njit(inline="always")
def compute_threshold_cost(
    relative_margin: float,
    optimal_cost: float,
    fx_q25: float,
    fx_q50: float,
    fx_q75: float,
    _c_opt: float = 1.0,
    _c_median: float = 1e-3,
    _c_range: float = 1e-3,
) -> float:
    """
    Compute threshold_cost=optimal_cost+cost_margin for curve fitting, to be used to determine which solutions are
    considered to be 'acceptable', which in turn is used to determine uncertainty ranges on parameters
    and derivative properties.

    The 'cost_margin' is computed based on 3 elements, each of which contribute to the threshold cost additively
      - optimal cost:     the worse the optimal fit, the more noisy / non-ideal the data is, and hence more uncertainty
      - median fx:        contributes a fixed amount proportional to overall magnitude of fx-values, even if optimal cost = 0
      - interquartile fx: contributes a fixed amount proportional to spread of fx-values, even if optimal cost = 0
                            --> this is complementary to using median fx, since the spread can vary vastly.
                                --> if exponent is very close to 0, spread is much lower than median
                                --> if exponent is large, spread is much higher than median and uncertainty range tends
                                       to be too small

    :param relative_margin: (float) relative size of the cost_margin
    :param optimal_cost: (float) cost of the optimal solution  (as returned by fitting_cost)
    :param fx_q25: (float) 25%-quantile of fx-values
    :param fx_q50: (float) 50%-quantile of fx-values
    :param fx_q75: (float) 75%-quantile of fx-values
    :param _c_opt: (float, default=1.0) coefficient of contribution of optimal_cost to cost_margin
    :param _c_median: (float, default=1e-3) coefficient of contribution of fx-median to cost_margin
    :param _c_range: (float, default=1e-3) coefficient of contribution of fx-range to cost_margin
    :return:
    """

    cost_margin_1 = _c_opt * optimal_cost
    cost_margin_2 = _c_median * abs(fx_q50)
    cost_margin_3 = _c_range * (fx_q75 - fx_q25)
    return optimal_cost + relative_margin * (cost_margin_1 + cost_margin_2 + cost_margin_3)
