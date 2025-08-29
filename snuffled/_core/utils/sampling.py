import random

import numba
import numpy as np


# =================================================================================================
#  Multi-scale sampling
# =================================================================================================
@numba.njit
def multi_scale_samples(x_min: float, x_max: float, dx_min: float, n: int, seed: int = 42) -> np.ndarray:
    """
    Returns n 'multiscale' samples across interval [x_min, x_max], with minimum distance between samples 'dx_min'.

    Distribution of distances between subsequent samples will be uniformly distributed (on a log-scale)
    in the interval [dx_min, dx_max], with dx_max computed in [(x_max-x_min)/n, (x_max-x_min)].

    At the same time, samples are spread 'evenly' (in a loose sense) across interval [x_min, x_max] with alternating
    densely and sparsely sampled regions.  Specific orders of interval widths is determined by randomization,
    to add a stochastic component to the sampling process (avoiding interaction with regular patterns in
    the functions we sample).

    Samples are guaranteed to be unique, sorted and will exactly include end-points x_min, x_max.

    Results across multiple runs are deterministic, but can be controlled by a seed.
    """

    # --- determine interval widths -----------------------

    # get interval widths   (n samples -> n-1 sub-intervals)
    n_w = n - 1
    w = get_fixed_sum_exponential_intervals(n=n_w, tgt_sum=x_max - x_min, dx_min=dx_min)

    # shuffle randomly
    np.random.seed(seed)
    np.random.shuffle(w)

    # --- determine actual samples ------------------------

    # compute cumulative
    x_rel = np.cumsum(w)  # x-positions of samples relative to x_min
    x_rel = x_rel * (x_max - x_min) / x_rel[-1]  # rescale (probably very slightly) to adjust for numerical errors

    # final samples
    x = np.zeros(n)
    x[0] = x_min  # make sure this exactly matches
    x[1:] = x_min + x_rel
    x[-1] = x_max  # make sure this exactly matches

    # we're done
    return x


# =================================================================================================
#  Exponential spacing helpers
# =================================================================================================
@numba.njit
def get_fixed_sum_exponential_intervals(n: int, tgt_sum: float, dx_min: float) -> np.ndarray:
    """
    Similar to fit_fixed_sum_exponential_intervals(...) but returns numpy array with actual interval sizes,
    instead of just factor c.
    """
    c = fit_fixed_sum_exponential_intervals(n, tgt_sum, dx_min)
    return dx_min * (c ** np.linspace(0, n - 1, n))


@numba.njit
def fit_fixed_sum_exponential_intervals(n: int, tgt_sum: float, dx_min: float) -> float:
    """
    PROBLEM STATEMENT

        This function tries to split an interval of size 'tgt_sum' (=target sum of sub-interval sizes) into 'n'
        sub-intervals, the sizes of which grow exponentially starting from 'dx_min' with a fixed scaling factor.

        Given that 'n', 'tgt_sum' and 'dx_min' are given, there is only one way to solve this problem.
        This method does that and returns the scaling factor c.

    MATHEMATICAL FORMULATION

        Mathematically speaking, the problem boils down to...

        Find a[i] (i=0,...,n-1) and c such that...
          - a[0] = dx_min
          - a[i+1] = c*a[i]
          - sum_i a[i] = tgt_sum

        Return c

    SOLUTION

        We can make use of the formula for the sum of exponential series

            sum_i    dx_min*(c^i)    =    dx_min * (1 - c^n) / (1-c)
           0...n-1

        Note that this formula is typically used for c<1, but also works for c>1 (which is this case).

        So we essentially need to solve for c:

            (1 - c^n) / (1-c) = tgt_sum / dx_min

        Note that 1 <= c <= (tgt_sum/dx_min)^(1/(n-1)), which gives us an interesting bisection starting interval.

    LIMITATIONS

        This problem is only well-defined if dx_min <= tgt_sum/n.  If not, we raise a ValueError

    """

    # --- argument handling -------------------------------
    if n < 2:
        raise ValueError(f"n should be >=2, here {n}")
    if dx_min <= 0.0:
        raise ValueError(f"we need dx_min > 0.0, here {dx_min}")
    if tgt_sum <= 0.0:
        raise ValueError(f"we need tgt_min > 0.0, here {tgt_sum}")
    if dx_min > tgt_sum / n:
        raise ValueError(f"we need dx_min <= tgt_sum/n, here {dx_min} > {tgt_sum}/{n}={tgt_sum / n}")
    elif dx_min == tgt_sum / n:
        return 1.0  # in this corner case we have c==1.0 as exact solution

    # --- solve -------------------------------------------
    rhs = tgt_sum / dx_min

    def f_bisect(_c: float) -> float:
        # this is the function for which we find a 0 using bisection
        return (1 - (_c**n)) / (1 - _c) - rhs

    c_min = 1.0
    c_max = rhs ** (1 / (n - 1))
    while True:
        c_mid = 0.5 * (c_min + c_max)
        if not c_min < c_mid < c_max:
            # iterate until we numerically cannot split [c_min, c_max] interval further
            break
        else:
            fc_mid = f_bisect(c_mid)
            if fc_mid == 0:
                # c_mid is spot on
                return c_mid
            elif fc_mid < 0:
                # c_mid is too small
                c_min = c_mid
            else:
                # c_mid is too large
                c_max = c_mid

    return c_mid


# =================================================================================================
#  Integer sampling
# =================================================================================================
def sample_integers(i_min: int, i_max: int, n: int, seed: int = 42) -> list[int]:
    """
    Sample n integers (without replacement) from range [i_min, i_max).
    Result is returned as a list of sorted integers.
    """
    random.seed(seed)
    return sorted(random.sample(range(i_min, i_max), n))


# =================================================================================================
#  Pseudo-uniform sampling
# =================================================================================================
@numba.njit
def pseudo_uniform_samples(x_min: float, x_max: float, n: int, seed: int = 42) -> np.ndarray:
    """
    Return n random numbers in interval [x_min, x_max), with 1 random sample in each of n uniform
    sub-intervals of width (x_max-x_min)/n.  Within each sub-interval uniform sampling is used.
    """

    # take care of corner cases
    if n < 0:
        raise ValueError("n should be >= 0")
    elif n == 0:
        return np.zeros(shape=(0,), dtype=np.float64)

    # prep
    np.random.seed(seed)
    interval_width = (x_max - x_min) / n
    interval_lefts = np.linspace(x_min, x_max - interval_width, n)  # left edges of each sub-interval

    # return
    return interval_lefts + interval_width * np.random.random(size=n)
