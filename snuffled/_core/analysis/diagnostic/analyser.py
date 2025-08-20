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
        props[Diagnostic.MAX_ZERO_WIDTH] = self._detect_max_zero_width()
        props[Diagnostic.NO_ZEROS_DETECTED] = self._detect_zeros_exist()
        props[Diagnostic.INTERVAL_NOT_BRACKETING_READY] = self._detect_interval_bracketing_ready()

        return props

    # -------------------------------------------------------------------------
    #  Internal snuffling methods
    # -------------------------------------------------------------------------
    def _detect_max_zero_width(self) -> float:
        return -1.0

    def _detect_zeros_exist(self) -> float:
        return -1.0

    def _detect_interval_bracketing_ready(self) -> float:
        return -1.0
