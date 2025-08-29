"""
Numba-specific helper functions, e.g. due to lack of support for existing ones, such as np.clip..
"""

import numpy as np

from snuffled._core.compatibility import numba


@numba.njit(inline="always")
def clip_scalar(a: float, a_min: float, a_max: float) -> float:
    """np.clip but just for scalars, compatible with numba."""
    if a < a_min:
        return a_min
    elif a > a_max:
        return a_max
    else:
        return a


@numba.njit(inline="always")
def geomean(x: np.ndarray) -> float:
    """Return geometric mean of array of positive numbers."""
    if len(x) == 0:
        return 1.0
    elif len(x) == 1:
        return float(x[0])
    else:
        x_min = np.min(x)
        if x_min < 0:
            raise ValueError(f"Cannot take geometric mean of negative numbers")
        elif x_min == 0.0:
            return 0.0
        else:
            return float(np.exp2(np.average(np.log2(x))))
