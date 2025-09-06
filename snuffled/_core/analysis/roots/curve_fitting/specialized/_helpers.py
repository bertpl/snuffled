import math

import numpy as np

from snuffled._core.compatibility import numba
from snuffled._core.utils.constants import EPS
from snuffled._core.utils.numba import clip_scalar, geomean

# pre-computed constant, to avoid unnecessary re-computation and to improve readability  (used in param_step(.))
__LN_R = math.log(2 * math.sqrt(2))  # ln(r) with r = 2*math.sqrt(2)


# =================================================================================================
#  Parameter initialization
# =================================================================================================
@numba.njit(inline="always")
def initialize_params(range_a: tuple[float, float], range_b: tuple[float, float], range_c: tuple[float, float]):
    """Return acceptable parameters within ranges"""

    # A: geometric mid-point of range (range_a values should always be positive)
    a = math.sqrt(range_a[0] * range_a[1])

    # B: 0 if possible within range
    b = clip_scalar(0.0, range_b[0], range_b[1])  # 0 if possible within range

    # C: 1 with same sign as range_c values (they always have same signs)
    c = clip_scalar(np.sign(range_c[0]) * 1.0, range_c[0], range_c[1])

    return a, b, c


# =================================================================================================
#  (a,b,c) search directions used in optimization & uncertainty exploration procedures
# =================================================================================================
@numba.njit
def param_step(
    a: float,
    b: float,
    c: float,
    method: str,
    step_size: float,
    range_a: tuple[float, float],
    range_b: tuple[float, float],
    range_c: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Take a step of size 'step_size' using method 'method' starting from current parameter values (a,b,c)
    and return (a_new, b_new, c_new).
    """

    # --- init --------------------------------------------
    a_new, b_new, c_new = a, b, c
    a_min, a_max = range_a
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
                # STEP 1: modify 'a' with a factor in [0.1, 10.0]
                a_new *= 10**step_size
                # STEP 2: modify 'c' if needed
                match method:
                    case "a":
                        # don't modify 'c', in this mode we only modify 'b'
                        pass
                    case "ac":
                        # INVARIANT: keep dg(r) - dg(1/r) constant, by adjusting c
                        #            with r=2*sqrt(2)    (=position of outermost x-value)
                        # since we keep b constant, this means we need to adjust c such that...
                        #    c = asinh(  (a'/a) * sinh(ln(r)*c') ) / ln(r)
                        ratio = a / a_new
                        c_new = math.asinh(ratio * math.sinh(__LN_R * c)) / __LN_R
            case "b" | "ba" | "bc":
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
                    case "ba":
                        # INVARIANT: keep dg(r) - dg(1/r) constant, by adjusting a
                        #            with r=2*sqrt(2)    (=position of outermost x-value)
                        # since we keep c constant, this means we need to adjust a such that...
                        #     a =  a'*(1-b')/(1-b)
                        ratio = (1 - b) / max(EPS, (1 - b_new))
                        a_new = a * ratio
                    case "bc":
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
        clip_scalar(float(a_new), a_min, a_max),
        clip_scalar(float(b_new), b_min, b_max),
        clip_scalar(float(c_new), c_min, c_max),
    )
