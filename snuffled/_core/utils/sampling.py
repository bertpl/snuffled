import math

import numpy as np

from snuffled._core.compatibility import numba


# =================================================================================================
#  Multi-scale sampling
# =================================================================================================
@numba.njit
def multi_scale_samples(x_min: float, x_max: float, dx_min: float, n: int, seed: int = 42) -> np.array:
    """
    Returns n 'multiscale' samples across interval [x_min, x_max], with minimum distance between samples 'dx_min'.

    Distribution of distances between subsequent samples will be uniformly distributed (on a log-scale)
    in the interval [dx_min, dx_max], with dx_max computed in [(x_max-x_min)/n, (x_max-x_min)].

    At the same time, samples are spread 'evenly' (in a loose sense) across interval [x_min, x_max] with alternating
    densely and sparsely sampled regions.  Specific orders of interval widths is partly determined by randomization,
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

    # determine sub-range count and sizes
    n_sub_ranges = 2 * int(math.sqrt(n))
    n_w_per_sub_range_mean = n_w / n_sub_ranges
    sub_range_sizes = get_fixed_sum_integers(
        n=n_sub_ranges,
        v_min=round(0.5 * n_w_per_sub_range_mean),
        v_max=round(1.5 * n_w_per_sub_range_mean),
        tgt_sum=n_w,
    )

    # reorder within sub-ranges
    #  - even subranges will have increasing sub-interval widths
    #  - odd subranges will have decreasing sub-interval widths
    i_from = 0  # inclusive
    for i_sub_range, sub_range_size in enumerate(sub_range_sizes):
        i_to = i_from + sub_range_size  # exclusive
        if i_sub_range % 2 == 0:
            # even -> increasing interval widths
            w[i_from:i_to] = np.sort(w[i_from:i_to])
        else:
            # odd -> decreasing interval widths
            w[i_from:i_to] = np.sort(w[i_from:i_to])[::-1]
        i_from += sub_range_size

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
#  Constrained integer sampling
# =================================================================================================
@numba.njit
def get_fixed_sum_integers(n: int, v_min: int, v_max: int, tgt_sum: int, seed: int = 42) -> np.ndarray:
    """Returns n integers sampled from interval [v_min, v_max] summing to tgt_sum."""

    # sample initial values
    # rng = np.random.default_rng(seed=seed)
    np.random.seed(seed)
    v = np.random.randint(low=v_min, high=v_max + 1, size=n)

    # apply random corrections to make the sum match
    sum_v = sum(v)
    rem = sum_v - tgt_sum  # rem = remained = how much the sum is too large
    while rem != 0:
        i = np.random.randint(0, n)
        if rem > 0:
            # try to decrease v[i]
            if v[i] > v_min:
                v[i] -= 1
                rem -= 1
        elif rem < 0:
            # try to increase v[i]
            if v[i] < v_max:
                v[i] += 1
                rem += 1

    # return result
    return v


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
    # fc_min = f_bisect(c_min)
    # fc_max = f_bisect(c_max)
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
                # fc_min = fc_mid
            else:
                # c_mid is too large
                c_max = c_mid
                # fc_max = fc_mid

    return c_mid
