import math

import numpy as np

__LOG2_ATANH_025 = math.log2(math.atanh(0.25))
__LOG2_ATANH_075 = math.log2(math.atanh(0.75))

from snuffled._core.compatibility import numba


@numba.njit(inline="always")
def smooth_sign(x: float, inner_tol: float, outer_tol: float) -> float:
    """
    Implements a smooth version of np.sign where 2 tolerances are used to determine when and how fast to transition
    for 0 to ±1.

            x                smooth_sign

        >> +outer_tol            +1.0
           +outer_tol            +0.75
           +inner_tol            +0.25
            0.0                   0.0
           -inner_tol            -0.25
           -outer_tol            -0.75
        << -outer_tol            -1.0

    We assume 0 < inner_tol < 4*inner_tol <= outer_tol.

    If this is not the case, we correct this such that outer_tol = 4*inner_tol.
    """

    if x == 0:
        return 0
    elif (inner_tol == 0) and (outer_tol == 0):
        # regular np.sign is desired?
        return np.sign(x)
    else:
        # split off sign and add back later
        x_abs, x_sign = abs(x), np.sign(x)

        # process inner_tol or outer_tol
        if inner_tol == 0:
            # we know for sure that outer_tol cannot be 0 in this case, which we'd catch earlier
            inner_tol = 0.25 * outer_tol
        elif inner_tol >= 0.25 * outer_tol:
            # we know that inner_tol > 0, in which case also inner_tol > 0
            mid_tol = math.sqrt(inner_tol * outer_tol)
            inner_tol = 0.5 * mid_tol
            outer_tol = 2.0 * mid_tol

        # compute f(x_abs) = math.tanh(g(z)) with z=x_abs/inner
        # with g(z) = exp2(s(log2(z)))
        #  and s a spline function
        if x_abs >= 100 * outer_tol:
            # shortcut -> this will always be 1.0
            abs_result = 1.0
        else:
            # regular case
            log2_z = np.log2(x_abs / inner_tol)
            log2_t = np.log2(outer_tol / inner_tol)  # t = outer_tol/inner_tol

            c = (__LOG2_ATANH_075 - __LOG2_ATANH_025) / log2_t
            s = __LOG2_ATANH_025 + (c * log2_z) + (1 - c) * (_base_spline(log2_z - log2_t) - _base_spline(-log2_z))
            abs_result = math.tanh(np.exp2(s))

        # add sign again
        return x_sign * abs_result


@numba.njit
def smooth_sign_array(x: np.ndarray, inner_tol: np.ndarray, outer_tol: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = smooth_sign(x[i], inner_tol[i], outer_tol[i])
    return result


# =================================================================================================
#  Internal helpers
# =================================================================================================
@numba.njit(inline="always")
def _base_spline(_x: float) -> float:
    """
    smoothed RELU-like spline basis function:

                               0 for x <= 0
        quadratically increasing for x in [0,1]
             linear with slope 1 for x >= 1

    Continuously differentiable function.
    """
    if _x <= 0:
        return 0.0
    elif _x <= 1:
        return 0.5 * _x * _x
    else:
        return _x - 0.5
