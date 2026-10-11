import hashlib

import pytest

# Marks tests that only make sense with numba JIT active — slow whole-pipeline sweeps, or
# assertions about compiled behavior. Spelled as an importable name so a typo fails at
# collection under --strict-markers; the conftest hook auto-skips these when JIT is
# disabled (see tests/conftest.py for why the suite runs in two modes).
only_with_numba_jit = pytest.mark.only_with_numba_jit

__SCALE = (1 << 128) - 1  # max value of 128-bit hash


def is_sorted_with_tolerance(lst: list[float], abs_tol: float) -> bool:
    """Check if list of numbers is sorted up to an absolute tolerance."""
    return all(lst[i] <= lst[i + 1] + abs_tol for i in range(len(lst) - 1))


def noise_from_float(x: float) -> float:
    """Compute a noise value in [-1, 1] from a float value.

    The function is pseudo-random, but deterministic, based on a hash of the string representation of the float.
    """
    hex_hash = hashlib.md5(str(x).encode("utf-8"), usedforsecurity=False).hexdigest()
    int_hash = int(hex_hash, 16)

    return -1.0 + 2.0 * (int_hash / __SCALE)
