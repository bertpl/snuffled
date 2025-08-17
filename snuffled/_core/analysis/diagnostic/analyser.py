from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.models.properties import Diagnostic, SnuffledDiagnostics


class DiagnosticAnalyser:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_data: FunctionSampler):
        self.function_data = function_data

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def analyse(self) -> SnuffledDiagnostics:
        props = SnuffledDiagnostics()
        props[Diagnostic.SIGN_CHANGE_MISSING] = self._detect_sign_change_missing()
        props[Diagnostic.MAX_ZERO_WIDTH] = self._detect_max_zero_width()
        return props

    # -------------------------------------------------------------------------
    #  Internal snuffling methods
    # -------------------------------------------------------------------------
    def _detect_sign_change_missing(self) -> float:
        return -1.0

    def _detect_max_zero_width(self) -> float:
        return -1.0
