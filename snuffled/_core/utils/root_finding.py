from typing import Callable

import numpy as np

from snuffled._core.models.root_analysis import Root


def find_odd_root(fun: Callable[[float], float], x_min: float, x_max: float, dx_min: float) -> Root:
    """
    Finds a root of fun(x) in interval [x_min, x_max], assuming fun(x_min) * fun(x_max) < 0.

    NOTE: This function only returns ODD roots, i.e. with sign(fx_min) != sign(fx_max).  Use find_root(.) for
          finding any root, including even ones.

    :param fun: Function for which we want to find a root
    :param x_min: (float) left edge of interval [x_min, x_max] in which to search for root
    :param x_max: (float) right edge of interval [x_min, x_max] in which to search for root
    :param dx_min: (float > 0) smallest interval to consider when looking for a root or determining its width
    """

    # --- error checking ----------------------------------
    fx_min, fx_max = fun(x_min), fun(x_max)
    if np.sign(fx_min) * np.sign(fx_max) != -1.0:
        raise ValueError(
            "fun(x_min) and fun(x_max) should have opposite signs. "
            + f"Here sign(fun(x_min))={np.sign(fx_min)}, sign(fun(x_max))={np.sign(fx_max)}"
        )

    # --- main loop ---------------------------------------
    for _ in range(100):
        # We iteratively look for odd roots in an ever-narrowing interval.  Mostly we only need 1 iteration,
        # except when we find a root that ends up being even, in which case the current interval must also contain other
        # (and odd) roots, in which case we look further in the appropriate remaining interval left or right of the
        # discovered even root.
        #
        # INVARIANT: at the beginning of this loop we have x_min < x_max and sign(fx_min) = -sign(fx_max).
        root = find_root(fun, x_min, x_max, dx_min)

        if root.deriv_sign != 0:
            # this is an ODD root, we can return this one
            return root
        else:
            # this is an EVEN root, we should look further in a sub-interval
            if np.sign(root.fx_min) != np.sign(fx_min):
                # --> [x_min, root.x_min] is an appropriate interval to look further
                x_max = root.x_min
                fx_max = root.fx_min
                print(f"Found EVEN root.  Continuing search in [{x_min},{x_max}]")
            else:
                # in this case we know sign(fx_min) == sign(root.fx_min) == sign(root.fx_max) != sign(fx_max).
                # --> [root.x_max, x_max] is an appropriate interval to look further
                x_min = root.x_max
                fx_min = root.fx_max
                print(f"Found EVEN root.  Continuing search in [{x_min},{x_max}]")

    # --- edge case ---------------------------------------
    # We can end up in this case if after 100 tries we only found even roots.  This can only reasonably happen
    # in extreme edge cases, in which case we'll return the most reasonable root, which potentially is too wide.
    # Since the analyses offered by this package are intended to be runnable at scale, fully automatically,
    # we want to avoid infinite loops at all cost.
    return Root(
        x_min=x_min,
        x=root.x,
        x_max=x_max,
        fx_min=fx_min,
        fx=root.fx,  # we know this is 0.0, since 'root' represents an even root
        fx_max=fx_max,  # we know this is of opposite sign as fx_min
    )


def find_root(fun: Callable[[float], float], x_min: float, x_max: float, dx_min: float) -> Root:
    """
    Finds a root of fun(x) in interval [x_min, x_max], assuming fun(x_min) * fun(x_max) < 0.

    NOTE 1: This function can return both ODD or EVEN roots, i.e. with derig_sign!=0 or deriv_sign==0.
            Use find_odd_root(.) for finding any root, including even ones.

    NOTE 2: We make sure we try to identify the root width (x_max-x_min) tightly.

    :param fun: Function for which we want to find a root
    :param x_min: (float) left edge of interval [x_min, x_max] in which to search for root
    :param x_max: (float) right edge of interval [x_min, x_max] in which to search for root
    :param dx_min: (float > 0) smallest interval to consider when looking for a root or determining its width
    """

    # --- error checking ----------------------------------
    fx_min, fx_max = fun(x_min), fun(x_max)
    if np.sign(fx_min) * np.sign(fx_max) != -1.0:
        raise ValueError(
            "fun(x_min) and fun(x_max) should have opposite signs. "
            + f"Here sign(fun(x_min))={np.sign(fx_min)}, sign(fun(x_max))={np.sign(fx_max)}"
        )

    # --- bisection ---------------------------------------
    orig_x_min, orig_x_max = x_min, x_max
    while True:
        # sample mid-point
        x_mid = 0.5 * (x_max + x_min)
        fx_mid = fun(x_mid)

        # decide how to go forward
        if fx_mid == 0.0:
            # found an exact root -> determine width & return
            return determine_root_width(fun, x_mid, orig_x_min, orig_x_max, dx_min)
        elif (x_mid == x_min) or (x_mid == x_max) or (x_max - x_min <= dx_min):
            # fx_min & fx_max are so close to each other that fx_mid coincides with either due to rounding
            # OR interval width is <= dx_min
            return Root(x_min, x_mid, x_max, fx_min, fx_mid, fx_max)
        elif np.sign(fx_mid) == np.sign(fx_min):
            x_min = x_mid
            fx_min = fx_mid
        else:
            x_max = x_mid
            fx_max = fx_mid


def determine_root_width(fun: Callable[[float], float], root: float, x_min: float, x_max: float, dx_min: float) -> Root:
    """
    Determine the 'width' of a function root.  This function is only used in case find_root finds a root with
    fun(root)==0.0.  In this case we try to find the largest interval around root (starting from interval size dx_min)
    for which fun(x)==0.0.

    This is a critical property to detect, since this determines how 'well-defined' the roots of the function are.
    If a root is considered 'wide' (i.e. order of magnitude close to the root-finding abs tolerance) they can start to
    affect root-finding efficiency (in a way that might benefit some algorithms).

    Also, this approach helps us to determine in which 'direction' the function is evolving (upward or downward)
    through the root, which is useful info for downstream analyses.

    This function never calls 'fun' with arguments outside [x_min, x_max], which also inevitably limits the width
    detection of roots very close to these interval edges.

    NOTE: we assume fun(x_min) and fun(x_max) are !=0 and have opposite signs, which is guaranteed by how find_root
          calls this function.
    """

    # --- validation --------------------------------------
    if (root == x_min) or (root == x_max):
        raise ValueError("We expect x_min < root < x_max, since we assume f(x_min)!=0, f(root)==0, f(x_max)!=0.")
    if fun(root) != 0.0:
        raise ValueError(f"We expect fun(root)==0.0, here {fun(root)}.")
    if np.sign(fun(x_min)) * np.sign(fun(x_max)) != -1.0:
        raise ValueError(f"We expect fun(x_min) and fun(x_max) to have opposite signs.")

    # --- init --------------------------------------------
    dx_start = max(
        dx_min / 2.0,  # taking a step of (dx_min/2) in two directions creates an interval of size dx_min
        min(
            np.nextafter(root, np.inf) - root,  # smallest step in negative direction
            root - np.nextafter(root, -np.inf),  # smallest step in positive direction
        ),
    )

    # --- search in + direction ---------------------------
    dx = dx_start
    while True:
        root_max = min(x_max, root + dx)  # once we reach root_max==x_max, we know fun(root_max) != 0
        froot_max = fun(root_max)
        if froot_max != 0.0:
            # we reached the edge of the root
            break
        else:
            # fun(root_max) is still 0.0, so we continue increasing dx
            dx += max(dx_start, 0.1 * dx)  # increment dx in steps of 10% or dx_start (whichever is largest)

    # --- search in - direction ----------------------------
    dx = dx_start
    while True:
        root_min = max(x_min, root - dx)  # once we reach root_min==x_min, we know fun(root_min) != 0
        froot_min = fun(root_min)
        if froot_min != 0.0:
            # we reached the edge of the root
            break
        else:
            # fun(root_min) is still 0.0, so we continue increasing dx
            dx += max(dx_start, 0.1 * dx)  # increment dx in steps of 10% or dx_start (whichever is largest)

    # construct Root object & return
    return Root(
        x_min=root_min,
        x=root,
        x_max=root_max,
        fx_min=froot_min,
        fx=0.0,
        fx_max=froot_max,
    )
