import math

import numba
import numpy as np

from snuffled._core.utils.numba import clip_scalar

from ._curves_and_costs import fitting_cost


# =================================================================================================
#  Find solution WITH uncertainty
# =================================================================================================
def fit_curve_with_uncertainty_tailored():
    pass


# =================================================================================================
#  Find solution WITHOUT uncertainty - OPTIMAL FIT
# =================================================================================================
@numba.njit
def fit_curve_tailored(
    x: np.ndarray,
    fx: np.ndarray,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
    reg: float,
) -> tuple[float, float, float]:
    """
    Tailored approach to find approximately optimal initial parameters (according to 'fitting_cost' method)
    with provided bounds of b and c parameters.

    We assume that (x, fx) have been generated using the compute_x_deltas method and that x-values have been
    rescaled such that median(x) == 1.0, i.e. by dividing by 2*dx.

    :return: (a,b,c)-tuple of parameters that are considered optimal, with b & c in requested ranges.
    """

    # --- initial point -----------------------------------------
    pass

    # --- refine ------------------------------------------------
    pass


# =================================================================================================
#  Find solution WITHOUT uncertainty - EXACT
# =================================================================================================
@numba.njit
def fit_curve_exact_three_points(
    fx_05: float,
    fx_1: float,
    fx_2: float,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Compute a,b,c exactly based on 3 data points  (0.5, fx_05), (1.0, fx_1), (2.0, fx_2),
    with solution being guaranteed to lie inside imposed b & c ranges.

    Reminder, we're trying to fit this function:

        g(x) = a*(b + (1-b)*(x^c))

    Solution is returned as tuple (a,b,c).
    """

    # initialize
    b_min, b_max = range_b
    c_min, c_max = range_c

    # computing a is straightforward
    a = fx_1
    if a == 0.0:
        # we cannot infer anything about the shape -> assume linear (c=1) without discontinuity (b=0)
        return a, 0.0, 1.0

    # Computing c (the exponent) essentially looks at the ratio of (fx_2-fx_1)/(fx_1-fx_05), which is
    # expected to be 2.0 if the function is linear, larger if c>1 and smaller of c<1.
    # Considering this ratio of differences also conveniently rids us of the influence of parameter b,
    # which we haven't determined yet.
    if fx_1 == fx_05:
        # ill conditioned, might be a sign of c being very close to 0, which squishes all f(x) value together
        c = c_min
    else:
        ratio = (fx_2 - fx_1) / (fx_1 - fx_05)
        if ratio <= 0:
            # this might again be a sign of poor conditioning due to all f(x) being squished together
            c = c_min
        else:
            c = clip_scalar(np.log2(ratio), c_min, c_max)

    # Now we can compute b by simply filling in a, c and solving for b
    b = 1 - (fx_2 - fx_05) / (fx_1 * (np.exp2(c) - np.exp2(-c)))
    b = clip_scalar(b, b_min, b_max)

    # return results as a tuple
    return a, b, c
