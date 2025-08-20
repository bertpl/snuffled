import numpy as np

from snuffled._core.compatibility import numba


@numba.njit
def non_monotonicity_score(fx_diff: np.ndarray, fx_diff_signs: np.ndarray) -> float:
    """
    Computes the non-monotonicity score, by computing the equally weighted mean of the following 3 scores:
      - up/down ratio in terms of f(x)       (equal up and down = 1.0)
      - up/down ratio in terms of x          (equal up and down = 1.0)
      - number of up/down flips              (>=50% of samples have flips = 1.0)

    Note that it is perfectly possible for this score to be 1.0 or very, very close, e.g. in case of an extremely noisy
    function.
    :param fx_diff: np.diff(fx_values)
    :param fx_diff_signs: smooth_sign_array(np.diff(fx_values))
    :return: (float) score in [0, 1]
    """

    return (
        non_monotonicity_score_up_down_fx(fx_diff)
        + non_monotonicity_score_up_down_x(fx_diff_signs)
        + non_monotonicity_score_n_up_down_flips(fx_diff_signs)
    ) / 3


@numba.njit(inline="always")
def non_monotonicity_score_up_down_fx(fx_diff: np.ndarray) -> float:
    total_up_fx = sum(np.maximum(0, fx_diff))
    total_down_fx = sum(np.maximum(0, -fx_diff))
    if (total_up_fx == 0.0) or (total_down_fx == 0.0):
        return 0.0
    else:
        return float(min(total_up_fx, total_down_fx) / max(total_up_fx, total_down_fx))


@numba.njit(inline="always")
def non_monotonicity_score_up_down_x(fx_diff_signs: np.ndarray) -> float:
    total_up_x = sum(np.maximum(0, fx_diff_signs))
    total_down_x = sum(np.maximum(0, -fx_diff_signs))
    if (total_up_x == 0.0) or (total_down_x == 0.0):
        return 0.0
    else:
        return float(min(total_up_x, total_down_x) / max(total_up_x, total_down_x))


@numba.njit(inline="always")
def non_monotonicity_score_n_up_down_flips(fx_diff_signs: np.ndarray) -> float:
    """
    Analyse fx_diff_sign (based on smooth_sign function) and check how often this sign flips from a value
    <0 to >0 or vice versa.  Each time this happens, we count the smallest of the two extrema as contribution of that
    flip.

    The theoretical maximum we could obtain that way is (n-1), which would happen for sign sequence [-1, 1, -1, ...].
    In practice, even for perfectly noisy sequence (with random sign flips with 50% chance) we would expect only a
    max value of 0.5*(n-1), so we use this normalization factor to normalize our result to [0,1] (with extra clipping,
    just to be sure).

    Examples:

        [-1,  1, -1,  1, -1]                ->    1.0      (Because we clip to [0,1])
        [-1, -1,  1,  1, -1]                ->    1.0
        [-1,  1,  1,  1,  1]                ->    0.5
        [-0.5,  0.5,  1,    0.5,  -0.5]     ->    0.50
        [-0.5,  0.2,  0.5,  0.2,  -0.5]     ->    0.50
        [-0.2,  0.2, -0.2,  0.2,  -0.2]     ->    0.20
        [-1,  -0.1,   0.01, -0.1,  -1]      ->    0.01

    """
    extrema = [fx_diff_signs[0]]
    for fx_diff_sign in fx_diff_signs:
        if np.sign(fx_diff_sign) == np.sign(extrema[-1]):
            # same sign as last extrema remembered
            if abs(fx_diff_sign) > abs(extrema[-1]):
                # more extreme value of same sign -> overwrite last element
                extrema[-1] = fx_diff_sign
        else:
            # different sign as last extrema remembered -> append
            extrema.append(fx_diff_sign)

    if len(extrema) > 1:
        score = sum([min(abs(float(v1)), abs(float(v2))) for v1, v2 in zip(extrema[:-1], extrema[1:])])
        normalized_score = min(1.0, score / (0.5 * (len(fx_diff_signs) - 1)))
        return float(normalized_score)
    else:
        # no sign flips in fx_diff_signs
        return 0.0
