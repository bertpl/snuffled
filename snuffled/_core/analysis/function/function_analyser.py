import math

import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models import FunctionProperty, SnuffledFunctionProperties
from snuffled._core.utils.statistics import estimate_sign_flip_frequency

from .helpers_discontinuous import discontinuity_score
from .helpers_non_monotonic import non_monotonicity_score


class FunctionAnalyser(PropertyExtractor[SnuffledFunctionProperties]):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_sampler: FunctionSampler):
        super().__init__(function_sampler)

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def supported_properties(self) -> list[str]:
        return [
            FunctionProperty.HIGH_DYNAMIC_RANGE,
            FunctionProperty.MANY_ZEROES,
            FunctionProperty.FLAT_INTERVALS,
            FunctionProperty.NON_MONOTONIC,
            FunctionProperty.DISCONTINUOUS,  # this goes last to benefit from sampling of earlier properties
        ]

    def _new_named_array(self) -> SnuffledFunctionProperties:
        return SnuffledFunctionProperties()

    def _extract(self, prop: str) -> float:
        match prop:
            case FunctionProperty.HIGH_DYNAMIC_RANGE:
                return self._extract_high_dynamic_range()
            case FunctionProperty.MANY_ZEROES:
                return self._extract_many_zeroes()
            case FunctionProperty.FLAT_INTERVALS:
                return self._extract_flat_intervals()
            case FunctionProperty.NON_MONOTONIC:
                return self._extract_non_monotonic()
            case FunctionProperty.DISCONTINUOUS:
                return self._extract_discontinuous()
            case _:
                raise ValueError(f"Property {prop} not supported")

    # -------------------------------------------------------------------------
    #  Internal methods
    # -------------------------------------------------------------------------
    def _extract_high_dynamic_range(self) -> float:
        """
        Looks at q10, q90 percentiles of abs(f(x)) for x in [x_min, x_max] and
        looks at the ratio q90/q10.  High dynamic range score is calibrated as:

            q90/q10     score

            2^10         0.0     (most 'normal' functions will fall below)
            2^52         0.5     (= accuracy of 64-bit float)
            2^94         1.0     (extrapolated from the above)

        :return: score in [0.0, 1.0] indicating to what extent this function exhibits a high dynamic range.
        """
        q10 = self.function_sampler.fx_quantile(0.1, absolute=True)
        q90 = self.function_sampler.fx_quantile(0.9, absolute=True)
        return float(np.interp(np.log2(q90 / q10), [10.0, 94.0], [0.0, 1.0], left=0.0, right=1.0))

    def _extract_many_zeroes(self) -> float:
        """
        The MANY_ZEROES score indicates if we're 'suffering' from a large number of zeroes,
        and is calibrated on a log scale as follows:

            estimated # of zeroes            score

                1                             0.0
                (x_max-x_min)/dx              1.0

        We estimate the # of zeroes by using the 'estimate_sign_flip_frequency' stats method.

        :return: (float) score in [0,1]
        """

        # Estimate 'zero frequency' lambda = # of zeroes estimated to occur in a unit interval

        #  STEP 1: build list of observations (w, p) with w the interval width and p indicating if we have a sign flip
        root_intervals, non_root_intervals = self.function_sampler.candidate_root_intervals()
        observations = [(x_right - x_left, 1.0) for x_left, x_right in root_intervals]
        observations += [(x_right - x_left, 0.0) for x_left, x_right in non_root_intervals]
        w_values = np.array([w for w, p in observations])
        p_values = np.array([p for w, p in observations])

        #  STEP 2: estimate lambda
        # note that we use a factor of 1.1 in lambda_min to ensure we only give score>0.0 if we see some convincing
        # evidence for multiple zeroes.
        lambda_min = 1.1 / (self.function_sampler.x_max - self.function_sampler.x_min)
        lambda_max = 1 / self.function_sampler.dx
        zeroes_lambda = estimate_sign_flip_frequency(w_values, p_values, lambda_min, lambda_max)

        #  STEP 3: convert to score in [0,1]
        score = (math.log2(zeroes_lambda) - math.log2(lambda_min)) / (math.log2(lambda_max) - math.log2(lambda_min))
        score = min(max(score, 0.0), 1.0)

        # done
        return float(score)

    def _extract_non_monotonic(self) -> float:
        return non_monotonicity_score(
            fx_diff=self.function_sampler.fx_diff_values(),
            fx_diff_signs=self.function_sampler.fx_diff_smooth_sign(),
        )

    def _extract_flat_intervals(self) -> float:
        fx_sign = self.function_sampler.fx_diff_smooth_sign()
        return 1.0 - float(np.mean(abs(fx_sign)))

    def _extract_discontinuous(self) -> float:
        return discontinuity_score(
            function_sampler=self.function_sampler,
            n_samples=self.function_sampler.n_fun_samples,
        )
