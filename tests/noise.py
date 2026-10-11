import hashlib
from functools import cache

import numpy as np

__SCALE = (1 << 128) - 1  # max value of 128-bit hash


def noise_from_float(x: float) -> float:
    """Compute a noise value in [-1, 1] from a float value.

    The function is pseudo-random, but deterministic, based on a hash of the string representation of the float.
    """
    hex_hash = hashlib.md5(str(x).encode("utf-8"), usedforsecurity=False).hexdigest()
    int_hash = int(hex_hash, 16)

    return -1.0 + 2.0 * (int_hash / __SCALE)


@cache
def deterministic_noise_series(n: int) -> np.ndarray:
    """Return n noise values in [-1, 1], the same for every call with the same n.

    The array is cached and shared by every caller, so it is read-only: a test that changed it in place would
    change the input of every later test in the same process.
    """
    rng = np.random.default_rng(seed=42)
    values = rng.uniform(low=-1.0, high=1.0, size=n)
    values.flags.writeable = False
    return values
