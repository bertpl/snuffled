import hashlib
from functools import cache

import numpy as np

__SCALE = (1 << 128) - 1  # max value of 128-bit hash


def noise_from_float(x: float) -> float:
    """
    Computes a noise value in [-1, 1] from a float value.  The function is pseudo-random, but deterministic,
    based on a hash of the string representation of the float.
    """
    hex_hash = hashlib.md5(str(x).encode("utf-8")).hexdigest()
    int_hash = int(hex_hash, 16)

    return -1.0 + 2.0 * (int_hash / __SCALE)


@cache
def deterministic_noise_series(n: int) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.uniform(low=-1.0, high=1.0, size=n)
