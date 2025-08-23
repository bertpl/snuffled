import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models import FunctionProperty, SnuffledFunctionProperties

from .helpers_discontinuous import discontinuity_score
from .helpers_many_zeroes import compute_many_zeroes_score
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
        and is calibrated on a log-like scale as follows:

            # of zeroes detected
              using multiscale sampling                 score

                1                                        0.0
                3                                       ~0.1
                n_fun_samples/2  (*)                     1.0

        (8) For functions with >>n_fun_samples zeroes, an interval having a sign switch becomes a stochastic process
            with 50% chance of detecting a zero.  Hence, n_fun_samples/2 is the highest stochastically expected value.

        :return: (float) score in [0,1]
        """
        return compute_many_zeroes_score(
            n=len(self.function_sampler.candidate_root_intervals()),
            n_max=int(self.function_sampler.n_fun_samples / 2.0),
        )

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
