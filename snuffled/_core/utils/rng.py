"""This module draws random numbers from a caller-owned state array, for numba-compiled functions.

The generator is xoroshiro128+ (Blackman and Vigna, https://prng.di.unimi.it/), seeded through
splitmix64. Its state is a 2-element uint64 array that every draw updates in place: a function that
draws random numbers takes that array as an argument, so no global state is involved.

The functions return the same numbers with numba's JIT on and off.

Every function declares its numba signature, so that the uint64 arithmetic is typed explicitly: a
call with arguments that do not cast to the declared types raises a TypeError, without compiling
another version. numba compiles each function when its decorator runs, at import time, so the
functions are defined below the functions that they call.

The uint64 arithmetic wraps around on overflow by design; without the JIT, NumPy warns about each
wraparound.

The code is a copy of the functions that snuffled needs from the max-div package (version 0.21.5,
module `max_div._core._random._rng`).
"""

import numba
import numpy as np

# ==================================================================================================
#  Constants
# ==================================================================================================
# The scale factor maps 2**64 to 1.0 - eps, so that every float64 draw lies below 1.0.
_UINT64_TO_UNIT_INTERVAL_SCALE = np.float64((1.0 - np.finfo(np.float64).eps) / (2**64))


# ==================================================================================================
#  Helpers
# ==================================================================================================
@numba.njit("uint64(uint64, uint64)", inline="always")
def _rotate_bits_left(x: np.uint64, k: np.uint64) -> np.uint64:
    """Rotate the 64 bits of `x` left by `k` positions; `k` must lie in 1..63."""
    return (x << k) | (x >> (np.uint64(64) - k))


@numba.njit("uint64(uint64[:])", inline="always")
def _xoroshiro128plus_next(rng_state: np.ndarray) -> np.uint64:
    """Return the next uint64 of the xoroshiro128+ sequence, advancing `rng_state` in place."""
    s0 = rng_state[0]
    s1 = rng_state[1]
    result = s0 + s1

    s1 ^= s0
    rng_state[0] = _rotate_bits_left(s0, np.uint64(24)) ^ s1 ^ (s1 << np.uint64(16))
    rng_state[1] = _rotate_bits_left(s1, np.uint64(37))

    return result


@numba.njit("uint64(uint64[:])", inline="always")
def _splitmix64_next(splitmix64_state: np.ndarray) -> np.uint64:
    """Return the next uint64 of the splitmix64 sequence, advancing the 1-element `splitmix64_state` in place."""
    z = splitmix64_state[0] + np.uint64(0x9E3779B97F4A7C15)
    splitmix64_state[0] = z
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


# ==================================================================================================
#  State
# ==================================================================================================
@numba.njit("uint64[:](int64)", inline="always")
def new_rng_state(seed: int) -> np.ndarray:
    """Return a new xoroshiro128+ state, derived from `seed` with splitmix64.

    `seed` must fit in an int64. A negative seed is reinterpreted as the uint64 with the same 64-bit
    pattern, so the state is the same with the JIT on and off.
    """
    # Without the JIT, NumPy refuses to convert a negative int to uint64 directly, so the seed goes
    # through int64 and is then viewed as uint64.
    splitmix64_state = np.array([seed], dtype=np.int64).view(np.uint64)
    rng_state = np.empty(2, dtype=np.uint64)
    rng_state[0] = _splitmix64_next(splitmix64_state)
    rng_state[1] = _splitmix64_next(splitmix64_state)
    return rng_state


# ==================================================================================================
#  Draws
# ==================================================================================================
@numba.njit("float64(uint64[:])", inline="always")
def rand_float64(rng_state: np.ndarray) -> float:
    """Return a random float64 in [0.0, 1.0), advancing `rng_state` in place."""
    return np.float64(_xoroshiro128plus_next(rng_state)) * _UINT64_TO_UNIT_INTERVAL_SCALE


@numba.njit("int64(uint64[:], int64, int64)", inline="always")
def rand_int64(rng_state: np.ndarray, low: int, high: int) -> np.int64:
    """Return a random integer in [low, high), advancing `rng_state` in place.

    `high` must exceed `low`. The draw reduces a uint64 modulo the range size, so each value's
    probability differs from 1 / (high - low) by a relative amount of at most (high - low) / 2**64.
    """
    range_size = np.uint64(high - low)
    return low + np.int64(_xoroshiro128plus_next(rng_state) % range_size)


@numba.njit("void(uint64[:], float64[:])", inline="always")
def shuffle_in_place(rng_state: np.ndarray, values: np.ndarray) -> None:
    """Shuffle the float64 array `values` in place with the Fisher–Yates algorithm, drawing from `rng_state`."""
    n = len(values)
    for i in range(n - 1):
        j = rand_int64(rng_state, i, n)
        values[i], values[j] = values[j], values[i]
