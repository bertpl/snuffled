from typing import Callable

import numpy as np


def find_root_and_width(
    fun: Callable[[float], float], x_min: float, x_max: float, dx_min: float
) -> tuple[float, float]:
    """
    Combines find_root() and determine_root_width(), such that in all cases we determine the root 'range'
    as [root_min, root_max] in a reliable way.

    :param fun: Function for which we want to find a root
    :param x_min: (float) left edge of interval [x_min, x_max] in which to search for root
    :param x_max: (float) right edge of interval [x_min, x_max] in which to search for root
    :param dx_min: (float > 0) smallest interval to consider when looking for a root or determining its width
    :return: (root_min, root_max)-tuple satisfying the following invariants:
                A) x_min <= root_min <= root_max <= x_max
                B) either of the following
                    - fun(root_min) == fun(root_max) == 0.0    in which case [root_min, root_max] is the largest interval for which this holds
                         OR
                    - fun(root_min) * fun(root_max) < 0.0  and [root_min, root_max] is the smallest possible interval >dx_min for which this holds

                    In both cases root_max-root_min gives an accurate idea of how well-defined (sharp vs wide) the root is.
    """
    root_min, root_max = find_root(fun, x_min, x_max, dx_min)
    if root_min == root_max:
        # this means fun(root_min)==fun(root_max)==0.0
        return determine_root_width(fun, root_min, x_min, x_max, dx_min)
    else:
        # this means fun(root_min) * fun(root_max) < 0.0
        return root_min, root_max


def find_root(fun: Callable[[float], float], x_min: float, x_max: float, dx_min: float) -> tuple[float, float]:
    """
    Find root in given interval with maximal precision up to the limit of absolute interval width 'dx_min'.
    The result is returned as an interval [root_min, root_max], which is always a sub-interval of [x_min, x_max].

    Either of the following conditions holds:
      - root_min == root_max  AND   fun(root_min) == fun(root_max) == 0.0
            OR
      - root_min < root_max   AND   fun(root_min) * fun(root_max) < 0.0
    """

    # --- sanity checks & initialization ------------------
    if x_min > x_max:
        raise ValueError(f"We expect x_min <= x_max; here {x_min} > {x_max}.")
    fx_min, fx_max = fun(x_min), fun(x_max)
    if fx_min == 0.0:
        return x_min, x_min
    elif fx_max == 0.0:
        return x_max, x_max
    elif np.sign(fx_min) == np.sign(fx_max):
        raise ValueError(f"No sign change in interval [{x_min}, {x_max}]; f(x_min)*f(x_max) = {fx_min * fx_max}")

    # --- bisection ---------------------------------------
    while (x_max - x_min) > dx_min:
        # sample mid-point
        x_mid = 0.5 * (x_max + x_min)
        fx_mid = fun(x_mid)

        # decide how to go forward
        if fx_mid == 0.0:
            # found the exact root
            return x_mid, x_mid
        elif (x_mid == x_min) or (x_mid == x_max):
            # fx_min & fx_max are so close to each other that fx_mid coincides with either due to rounding
            return x_min, x_max
        elif np.sign(fx_mid) == np.sign(fx_min):
            x_min = x_mid
            fx_min = fx_mid
        else:
            x_max = x_mid
            fx_max = fx_mid

    # return mid-point if we haven't returned prematurely yet
    return x_min, x_max


def determine_root_width(
    fun: Callable[[float], float], root: float, x_min: float, x_max: float, dx_min: float
) -> tuple[float, float]:
    """
    Determine the 'width' of a function root.  This function is only used in case find_root finds a root with
    fun(root)==0.0.  In this case we try to find the largest interval around root (starting from interval size dx_min)
    for which fun(x)==0.0.

    This is a useful property to detect, since this determines how 'well-defined' the roots of the function are.
    If they are considered 'wide' (i.e. order of magnitude close to the root-finding abs tolerance) they can start to
    affect root-finding efficiency (in a way that might benefit some algorithms).

    This function never calls 'fun' with arguments outside [x_min, x_max], which also inevitably limits the width
    detection of roots very close to these interval edges.

    NOTE: in case 'find_root' finds a root where root_min < root_max and hence fun(root_min) * fun(root_max) < 0,
          it means it couldn't find an exact root within the smallest achievable interval size.
          In other words [root_min, root_max] already represents the smallest possible interval size enclosing the root.
          This could either be due to the fact that numerical rounding artifacts cause the function to jump from <0 to
          >0 with a minimal change in x (as allowed by the float64 format), or because the function fundamentally
          makes a jump around the 'root' (e.g. step(x)-0.5).
    """

    # --- init --------------------------------------------
    dx_start = max(
        dx_min / 2.0,  # taking a step of (dx_min/2) in two directions creates an interval of size dx_min
        min(
            np.nextafter(root, np.inf) - root,  # smallest step in negative direction
            root - np.nextafter(root, -np.inf),  # smallest step in positive direction
        ),
    )
    root_min = root  # left-most edge of interval where fun(x)==0
    root_max = root  # right-most edge of interval where fun(x)==0

    # --- search in + direction ---------------------------
    dx = dx_start
    while root_max < x_max:
        x_cand = min(x_max, root + dx)
        if fun(x_cand) != 0.0:
            # we reached the edge of the root
            break
        else:
            # fun(x) is still 0.0, so we continue increasing dx
            root_max = x_cand
            dx += max(dx_start, 0.1 * dx)  # increment dx in steps of 10% or dx_start (whichever is largest)

    # --- search in - direction ----------------------------
    dx = dx_start
    while root_min > x_min:
        x_cand = max(x_min, root - dx)
        if fun(x_cand) != 0.0:
            # we reached the edge of the root
            break
        else:
            root_min = x_cand
            dx += max(dx_start, 0.1 * dx)  # increment dx in steps of 10% or dx_start (whichever is largest)

    # we're done
    return root_min, root_max
