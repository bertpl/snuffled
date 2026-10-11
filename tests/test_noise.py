import math

from tests.noise import noise_from_float


def test_noise_from_float():
    # --- arrange ----------------------
    n_values = 1_000
    x_values = [math.pi + math.e * i for i in range(n_values)]

    # --- act --------------------------
    noise_values = [noise_from_float(x) for x in x_values]

    # --- assert -----------------------
    assert min(noise_values) >= -1.0
    assert max(noise_values) <= 1.0
    assert len(set(noise_values)) == n_values
