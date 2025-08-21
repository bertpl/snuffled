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
    def _new_named_array(self) -> SnuffledDiagnostics:
        return SnuffledDiagnostics()

    def _extract(self, prop: str) -> float:
        match prop:
            case Diagnostic.INTERVAL_NOT_BRACKETING_READY:
                return self._extract_interval_not_bracketing_ready()
            case Diagnostic.NO_ZEROS_DETECTED:
                return self._extract_max_zero_width()
            case Diagnostic.INTERVAL_NOT_BRACKETING_READY:
                return self._extract_no_zeros_detected()
            case _:
                raise ValueError(f"Property {prop} not supported")

    # -------------------------------------------------------------------------
    #  Internal methods
    # -------------------------------------------------------------------------
    def _extract_interval_not_bracketing_ready(self) -> float:
        return -1.0

    def _extract_max_zero_width(self) -> float:
        return -1.0

    def _extract_no_zeros_detected(self) -> float:
        return -1.0
