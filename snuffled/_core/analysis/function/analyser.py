import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.compatibility import numba
from snuffled._core.models import FunctionProperty, SnuffledFunctionProperties


class FunctionAnalyser:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_data: FunctionSampler):
        self.function_data = function_data

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def analyse(self) -> SnuffledFunctionProperties:
        props = SnuffledFunctionProperties()
        props[FunctionProperty.MANY_ZEROES] = self._detect_many_zeroes()
        props[FunctionProperty.NON_MONOTONIC] = self._detect_non_monotonic()
        props[FunctionProperty.HIGH_DYNAMIC_RANGE] = self._detect_high_dynamic_range()
        props[FunctionProperty.DISCONTINUOUS] = self._detect_discontinuous()
        props[FunctionProperty.FLAT_INTERVALS] = self._detect_flat_intervals()
        return props

    # -------------------------------------------------------------------------
    #  Internal snuffling methods
    # -------------------------------------------------------------------------
    def _detect_many_zeroes(self) -> float:
        return -1.0

    def _detect_non_monotonic(self) -> float:
        return _non_monotonicity_score(
            fx_diff=self.function_data.fx_diff_values(),
            fx_diff_signs=self.function_data.fx_diff_smooth_sign(),
        )

    def _detect_high_dynamic_range(self) -> float:
        """
        Looks at q10, q90 percentiles of abs(f(x)) for x in [x_min, x_max] and
        looks at the ratio q90/q10.  High dynamic range score is calibrated as:

            q90/q10     score

            2^10         0.0     (most 'normal' functions will fall below)
            2^52         0.5     (= accuracy of 64-bit float)
            2^94         1.0     (extrapolated from the above)

        :return: score in [0.0, 1.0] indicating to what extent this function exhibits a high dynamic range.
        """
        q10 = self.function_data.fx_quantile(0.1, absolute=True)
        q90 = self.function_data.fx_quantile(0.9, absolute=True)
        return float(np.interp(np.log2(q90 / q10), [10.0, 94.0], [0.0, 1.0], left=0.0, right=1.0))

    def _detect_discontinuous(self) -> float:
        return -1.0

    def _detect_flat_intervals(self) -> float:
        fx_sign = self.function_data.fx_diff_smooth_sign()
        return 1.0 - float(np.mean(abs(fx_sign)))


# =================================================================================================
#  Helpers
# =================================================================================================
@numba.njit
def _non_monotonicity_score(fx_diff: np.ndarray, fx_diff_signs: np.ndarray) -> float:
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
        _non_monotonicity_score_up_down_fx(fx_diff)
        + _non_monotonicity_score_up_down_x(fx_diff_signs)
        + _non_monotonicity_score_n_up_down_flips(fx_diff_signs)
    ) / 3


@numba.njit(inline="always")
def _non_monotonicity_score_up_down_fx(fx_diff: np.ndarray) -> float:
    total_up_fx = sum(np.maximum(0, fx_diff))
    total_down_fx = sum(np.maximum(0, -fx_diff))
    if (total_up_fx == 0.0) or (total_down_fx == 0.0):
        return 0.0
    else:
        return min(total_up_fx, total_down_fx) / max(total_up_fx, total_down_fx)


@numba.njit(inline="always")
def _non_monotonicity_score_up_down_x(fx_diff_signs: np.ndarray) -> float:
    total_up_x = sum(np.maximum(0, fx_diff_signs))
    total_down_x = sum(np.maximum(0, -fx_diff_signs))
    if (total_up_x == 0.0) or (total_down_x == 0.0):
        return 0.0
    else:
        return min(total_up_x, total_down_x) / max(total_up_x, total_down_x)


@numba.njit(inline="always")
def _non_monotonicity_score_n_up_down_flips(fx_diff_signs: np.ndarray) -> float:
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
        return normalized_score
    else:
        # no sign flips in fx_diff_signs
        return 0.0
