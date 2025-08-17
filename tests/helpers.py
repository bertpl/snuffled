def is_sorted_with_tolerance(lst: list[float], abs_tol: float) -> bool:
    """Check if list of numbers is sorted up to an absolute tolerance."""
    return all(lst[i] <= lst[i + 1] + abs_tol for i in range(len(lst) - 1))
