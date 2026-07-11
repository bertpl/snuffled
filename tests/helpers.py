import pytest

# Marks tests that only make sense with numba JIT active — slow whole-pipeline sweeps, or
# assertions about compiled behavior. Spelled as an importable name so a typo fails at
# collection under --strict-markers; the conftest hook auto-skips these when JIT is
# disabled (see tests/conftest.py for why the suite runs in two modes).
only_with_numba_jit = pytest.mark.only_with_numba_jit


def is_sorted_with_tolerance(lst: list[float], abs_tol: float) -> bool:
    """Check if list of numbers is sorted up to an absolute tolerance."""
    return all(lst[i] <= lst[i + 1] + abs_tol for i in range(len(lst) - 1))
