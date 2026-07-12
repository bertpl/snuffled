from snuffled._core.utils.constants import MIN_USEFUL_N_SAMPLES


def require_min_n_samples(name: str, value: int) -> None:
    """Reject a sample-count parameter below MIN_USEFUL_N_SAMPLES, naming the offending parameter.

    Raises:
        ValueError: if ``value < MIN_USEFUL_N_SAMPLES``.
    """
    if value < MIN_USEFUL_N_SAMPLES:
        raise ValueError(f"{name} must be >= {MIN_USEFUL_N_SAMPLES}, got {value}")
