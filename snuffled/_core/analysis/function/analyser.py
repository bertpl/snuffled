import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
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
        return -1.0

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
        fx_sign = self.function_data.smooth_fx_sign()
        return 1.0 - float(np.mean(abs(fx_sign)))

        # fx_smooth_rel = self.function_data.fx_values(smoothing="relative")
        # fx_smooth_abs = self.function_data.fx_values(smoothing="absolute")
        #
        # dfx_smooth_rel = np.diff(fx_smooth_rel)
        # dfx_smooth_abs = np.diff(fx_smooth_abs)
        #
        # # the fewer unique values we have
        # local_flat_fraction = sum(dfx_smooth_rel == 0) / len(dfx_smooth_rel)
        # global_flat_fraction = sum(dfx_smooth_abs == 0) / len(dfx_smooth_abs)
        #
        # return 0.5 * (local_flat_fraction + global_flat_fraction)
