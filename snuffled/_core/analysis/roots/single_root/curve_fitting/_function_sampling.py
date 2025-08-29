import math

import numpy as np

from snuffled._core.compatibility import numba
from snuffled._core.utils.constants import SEED_OFFSET_COMPUTE_X_DELTAS
from snuffled._core.utils.sampling import pseudo_uniform_samples


@numba.njit
def compute_x_deltas(dx: float, k: int, seed: int) -> np.ndarray:
    """
    Computes x_delta-array, containing values that can be used to sample function values around
    a root:    f(root ± x_delta).

    x_delta values are constructed such that...
        - we will have 3*(1 + 2*k) values
             - 1 group around   dx
             - 1 group around 2*dx
             - 1 group around 4*dx
        - with each group consisting of...
             - 1 sample at the reference point
             - k samples below the reference point
             - k samples symmetrically (geomean) above the reference point
        - each of the 3 groups of samples will span a 2x range from 1/sqrt(2) -> sqrt(2) relative around the ref. value
           - hence different groups do not overlap
           - but they span a consecutive 8x range [dx/sqrt(2), 4*dx*sqrt(2)]

    This will result in delta_x values with the following properties:
        - median(x_deltas < dx*sqrt(2))    =  geomean(x_deltas < dx*sqrt(2))    =    dx
        - median(x_deltas)                 =  geomean(x_deltas)                 =  2*dx
        - median(x_deltas > 2*sqrt(2)*dx)  =  geomean(x_deltas > 2*sqrt(2)*dx)  =  4*dx

    :param dx: (float) reference distance dx > 0
    :param k: (int) sampling count parameter
    :param seed: (int) seed for random number generator
    :return: np.ndarray with 3*(1+2k) x_delta values, sorted in increasing order
    """

    # seed handling
    seed += SEED_OFFSET_COMPUTE_X_DELTAS

    # initialize
    x_deltas = np.zeros(3 + 6 * k)
    x_deltas[0] = dx
    x_deltas[1] = 2 * dx
    x_deltas[2] = 4 * dx

    # collect appropriate random numbers
    # (=random values in [1, sqrt(2))
    rand_values_outer = np.exp2(pseudo_uniform_samples(0.0, 0.5, k - 1, seed=seed))
    rand_values_inner = np.exp2(pseudo_uniform_samples(0.0, 0.5, k, seed=seed + 1))

    # 6*k additional randomized samples
    for j in range(k):
        # set up iteration j
        if j == 0:
            # make sure first outer samples are right at the edges,
            # which guarantees that we span the entire range [(2.0^-0.5)*dx, (2.0^2.5)*dx]
            r_outer = math.sqrt(2)
        else:
            r_outer = rand_values_outer[j - 1]
        r_inner = rand_values_inner[j]
        i_start = 3 + (6 * j)

        # two values geo-symmetrically around dx
        x_deltas[i_start] = dx * r_outer
        x_deltas[i_start + 1] = dx / r_outer

        # two values geo-symmetrically around 2*dx
        x_deltas[i_start + 2] = 2 * dx * r_inner
        x_deltas[i_start + 3] = 2 * dx / r_inner

        # two values geo-symmetrically around 4*dx
        x_deltas[i_start + 4] = 4 * dx * r_outer
        x_deltas[i_start + 5] = 4 * dx / r_outer

    return np.sort(x_deltas)
