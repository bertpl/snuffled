import math

from tests.helpers import noise_from_float


def test_noise_from_float():
    """noise_from_float maps distinct inputs to distinct values in [-1, 1]."""
    # --- arrange ----------------------
    n_values = 1_000
    x_values = [math.pi + math.e * i for i in range(n_values)]

    # --- act --------------------------
    noise_values = [noise_from_float(x) for x in x_values]

    # --- assert -----------------------
    assert min(noise_values) >= -1.0
    assert max(noise_values) <= 1.0
    assert len(set(noise_values)) == n_values
