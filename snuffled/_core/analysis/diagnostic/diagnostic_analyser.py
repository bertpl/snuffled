import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models.properties import Diagnostic, SnuffledDiagnostics


class DiagnosticAnalyser(PropertyExtractor[SnuffledDiagnostics]):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_sampler: FunctionSampler):
        super().__init__(function_sampler)

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def supported_properties(self) -> list[str]:
        # return in order of increasing number of required function evals
        return [
            Diagnostic.INTERVAL_NOT_BRACKETING_READY,
            Diagnostic.NO_ZEROS_DETECTED,
            Diagnostic.MAX_ZERO_WIDTH,
        ]

    def _new_named_array(self) -> SnuffledDiagnostics:
        return SnuffledDiagnostics()

    def _extract(self, prop: str) -> float:
        match prop:
            case Diagnostic.INTERVAL_NOT_BRACKETING_READY:
                return self._extract_interval_not_bracketing_ready()
            case Diagnostic.MAX_ZERO_WIDTH:
                return self._extract_max_zero_width()
            case Diagnostic.NO_ZEROS_DETECTED:
                return self._extract_no_zeros_detected()
            case _:
                raise ValueError(f"Property {prop} not supported")

    # -------------------------------------------------------------------------
    #  Internal methods
    # -------------------------------------------------------------------------
    def _extract_interval_not_bracketing_ready(self) -> float:
        x_min, x_max = self.function_sampler.x_min, self.function_sampler.x_max
        fx_min, fx_max = self.function_sampler.f(x_min), self.function_sampler.f(x_max)
        fx_min_sign, fx_max_sign = np.sign(fx_min), np.sign(fx_max)
        if fx_min_sign * fx_max_sign > 0:
            # interval end-point f-values have same sign -> NOT READY
            return 0.0
        elif fx_min_sign * fx_max_sign == 0.0:
            # one of the end-point f-values is 0         -> BORDERLINE
            return 0.5
        else:
            # end-point f-values have opposite sign      -> READY
            return 1.0

    def _extract_max_zero_width(self) -> float:
        return max([root_max - root_min for root_min, root_max in self.function_sampler.roots()])

    def _extract_no_zeros_detected(self) -> float:
        root_intervals, no_root_intervals = self.function_sampler.candidate_root_intervals()
        if len(root_intervals) == 0:
            # no candidate intervals to find roots
            return 1.0
        else:
            return 0.0
