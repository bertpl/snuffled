import math

import numpy as np

from snuffled._core.utils.noise import deterministic_noise_series, noise_from_float


def test_noise_from_float():
    # --- arrange -----------------------------------------
    n_values = 1_000
    x_values = [math.pi + math.e * i for i in range(n_values)]

    # --- act ---------------------------------------------
    noise_values = [noise_from_float(x) for x in x_values]

    # --- assert ------------------------------------------
    assert min(noise_values) >= -1.0
    assert max(noise_values) <= 1.0
    assert len(set(noise_values)) == n_values


def test_deterministic_noise_series():
    # --- arrange -----------------------------------------
    n = 10_000

    # --- act ---------------------------------------------
    noise_1 = deterministic_noise_series(n)
    noise_2 = deterministic_noise_series(n)
    deterministic_noise_series.cache_clear()  # to really make sure it is deterministic
    noise_3 = deterministic_noise_series(n)

    # --- assert ------------------------------------------
    assert np.array_equal(noise_1, noise_2)
    assert np.array_equal(noise_1, noise_3)
    assert len(noise_1) == n
    assert len(set(noise_1)) == n
    assert -1.0 <= min(noise_1) < -0.99 < 0.99 < max(noise_1) <= 1.0
