from typing import Callable

from .constants import EPS

# smallest interval we will consider for root-finding.
_DX_MIN = EPS * EPS


def find_root(fun: Callable[[float], float], x_min: float, x_max: float):
    """
    Find root in given interval with maximal precision up to the limit of absolute interval width _DX_MIN.
    The returned root x will either...
      - have f(x)==0.0 exactly
      - be the midpoint the smallest bracketing interval that shows a sign change.
    """

    # --- sanity checks & initialization ------------------
    if x_min > x_max:
        raise ValueError(f"We expect x_min <= x_max; here {x_min} > {x_max}.")
    fx_min, fx_max = fun(x_min), fun(x_max)
    if fx_min * fx_max > 0:
        raise ValueError(f"No sign change in interval [{x_min}, {x_max}]; f(x_min)*f(x_max) = {fx_min * fx_max}")
    elif fx_min == 0.0:
        return x_min
    elif fx_max == 0.0:
        return x_max

    # --- bisection ---------------------------------------
    while (x_max - x_min) > _DX_MIN:
        # sample mid-point
        x_mid = 0.5 * (x_max + x_min)
        fx_mid = fun(x_mid)

        # decide how to go forward
        if fx_mid == 0.0:
            # found the exact root
            return x_mid
        elif (x_mid == x_min) or (x_mid == x_max):
            # fx_min & fx_max are so close to each other that fx_mid coincides with either due to rounding
            return x_mid
        elif fx_mid * fx_min > 0:
            x_min = x_mid
            fx_min = fx_mid
        else:
            x_max = x_mid
            fx_max = fx_mid

    # return mid-point if we haven't returned prematurely yet
    return 0.5 * (x_max + x_min)
