import math

import numba
import numpy as np

from snuffled._core.utils.constants import EPS
from snuffled._core.utils.numba import clip_scalar, geomean

from ._curves_and_costs import fitting_cost, fitting_curve

# pre-computed constant, to avoid unnecessary re-computation and to improve readability
__LN_R = math.log(2 * math.sqrt(2))  # ln(r) with r = 2*math.sqrt(2),   used in param_step()


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


@numba.njit
def param_step(
    a: float,
    b: float,
    c: float,
    method: str,
    step_size: float,
    range_b: tuple[float, float],
    range_c: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Take a step of size 'step_size' using method 'method' starting from current parameter values (a,b,c)
    and return (a_new, b_new, c_new).
    """

    # --- init --------------------------------------------
    a_new, b_new, c_new = a, b, c
    b_min, b_max = range_b
    c_min, c_max = range_c

    # --- take step ---------------------------------------
    if step_size != 0.0:
        match method:
            case "a" | "ac":
                # -------------------------------
                # These steps first modify parameter 'a' in a certain way and then optionally modify 'c'
                # to satisfy an invariant
                # -------------------------------
                # STEP 1: modify 'a' with a factor in [0.5, 2.0]
                a_new *= np.exp2(step_size)
                # STEP 2: modify 'c' if needed
                match method:
                    case "a":
                        # don't modify 'c', in this mode we only modify 'b'
                        pass
                    case "ac_bal":
                        # INVARIANT: keep dg(r) - dg(1/r) constant, by adjusting c
                        #            with r=2*sqrt(2)    (=position of outermost x-value)
                        # since we keep b constant, this means we need to adjust c such that...
                        #    c = asinh(  (a'/a) * sinh(ln(r)*c') ) / ln(r)
                        ratio = a / a_new
                        c_new = math.asinh(ratio * math.sinh(__LN_R * c)) / __LN_R
            case "b" | "bc":
                # -------------------------------
                # These steps first modify parameter 'b' in a certain way and then optionally modify 'c'
                # to satisfy an invariant
                # -------------------------------
                # STEP 1: modify 'b' in [b_min, b_max] with step_size=-1 -> b_min and step_size=+1 -> b_max  (LIN scale)
                if step_size < 0:
                    b_new = b + step_size * (b - b_min)
                else:
                    b_new = b + step_size * (b_max - b)
                # STEP 2: modify 'c' if needed
                match method:
                    case "b":
                        # don't modify 'c', in this mode we only modify 'b'
                        pass
                    case "bc_bal":
                        # INVARIANT: keep dg(r) - dg(1/r) constant, by adjusting c
                        #            with r=2*sqrt(2)    (=position of outermost x-value)
                        # since we keep a constant, this means we need to adjust c such that...
                        #    c = asinh(  ((1-b')/(1-b)) * sinh(ln(r)*c') ) / ln(r)
                        ratio = (1 - b) / max(EPS, (1 - b_new))
                        c_new = math.asinh(ratio * math.sinh(__LN_R * c)) / __LN_R
            case "c":
                # modify 'c' with a factor in [0.5, 2.0]
                c_new *= np.exp2(step_size)
            case _:
                raise ValueError(f"Unknown step method: {method}")

    # --- return clipped updates --------------------------
    return (
        float(a_new),
        clip_scalar(float(b_new), b_min, b_max),
        clip_scalar(float(c_new), c_min, c_max),
    )


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
