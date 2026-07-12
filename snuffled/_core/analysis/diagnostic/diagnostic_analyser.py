import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models.properties import Diagnostic, SnuffledDiagnostics


class DiagnosticAnalyser(PropertyExtractor[SnuffledDiagnostics]):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_sampler: FunctionSampler) -> None:
        super().__init__(function_sampler)

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def supported_properties(self) -> list[str]:
        # return in order of increasing number of required function evals
        return [
            Diagnostic.NAN_VALUES_DETECTED,
            Diagnostic.INF_VALUES_DETECTED,
            Diagnostic.INTERVAL_NOT_BRACKETING_READY,
            Diagnostic.NO_ZEROS_DETECTED,
            Diagnostic.MAX_ZERO_WIDTH,
            Diagnostic.ALL_ROOTS_TOO_CLOSE_TO_EDGE,
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
            case Diagnostic.ALL_ROOTS_TOO_CLOSE_TO_EDGE:
                return self._extract_all_roots_too_close_to_edge()
            case Diagnostic.NAN_VALUES_DETECTED:
                return self._extract_nan_values_detected()
            case Diagnostic.INF_VALUES_DETECTED:
                return self._extract_inf_values_detected()
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
        if fx_min_sign * fx_max_sign == 0.0:
            # one of the end-point f-values is 0         -> BORDERLINE
            return 0.5
        # end-point f-values have opposite sign      -> READY
        return 1.0

    def _extract_max_zero_width(self) -> float:
        # default=0.0 for rootless functions: no roots -> no zero-width interval,
        # mirroring the RootsAnalyser no-roots contract (all root properties 0.0).
        return max((root.width for root in self.function_sampler.roots()), default=0.0)

    def _extract_no_zeros_detected(self) -> float:
        root_intervals, _no_root_intervals = self.function_sampler.candidate_root_intervals()
        if len(root_intervals) == 0:
            # no candidate intervals to find roots
            return 1.0
        return 0.0

    def _extract_all_roots_too_close_to_edge(self) -> float:
        # 1.0 iff roots exist but every one is too close to an interval edge to analyze reliably
        # (distinct from NO_ZEROS_DETECTED, which flags the genuinely rootless case).
        has_roots = len(self.function_sampler.roots()) > 0
        has_analyzable = len(self.function_sampler.analyzable_roots()) > 0
        return 1.0 if (has_roots and not has_analyzable) else 0.0

    def _extract_nan_values_detected(self) -> float:
        # 1.0 iff the function returned NaN for a sampled point (which the boundary sanitize replaced).
        # Force the multi-scale grid first so the flag is set even when extracted standalone.
        self.function_sampler.fx_values()
        return 1.0 if self.function_sampler.saw_nan else 0.0

    def _extract_inf_values_detected(self) -> float:
        # 1.0 iff the function returned +-inf for a sampled point (which the boundary clip replaced).
        self.function_sampler.fx_values()
        return 1.0 if self.function_sampler.saw_inf else 0.0
