import pytest

# Tier-B marker: tests that only make sense with numba JIT compilation active (slow
# whole-pipeline sweeps sized beyond what plain-Python execution can afford, or behavior
# specific to compiled code). Spelled as an importable name so a typo fails at collection
# time; conftest auto-skips marked tests when numba runs with JIT disabled.
only_with_numba_jit = pytest.mark.only_with_numba_jit


def is_sorted_with_tolerance(lst: list[float], abs_tol: float) -> bool:
    """Check if list of numbers is sorted up to an absolute tolerance."""
    return all(lst[i] <= lst[i + 1] + abs_tol for i in range(len(lst) - 1))
