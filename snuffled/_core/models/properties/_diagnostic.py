from enum import StrEnum

from snuffled._core.models.base import NamedArray


class Diagnostic(StrEnum):
    """Names of the diagnostics, which flag a function whose root-finding problem or analysis is not well-posed.

    Each diagnostic is a flag: 1.0 means the problem in its name is present, 0.0 that it is absent.
    INTERVAL_NOT_BRACKETING_READY also returns 0.5, when an interval end-point value is exactly 0.

    The exception is MAX_ZERO_WIDTH: the raw width, in x units, of the widest region around a root
    where the function is exactly 0, and 0.0 when there is no root. It is not a flag, so that callers
    can compare it against their own tolerance.
    """

    MAX_ZERO_WIDTH = "diagnostic_max_zero_width"
    NO_ZEROS_DETECTED = "diagnostic_no_zeros_detected"
    INTERVAL_NOT_BRACKETING_READY = "diagnostic_interval_not_bracketing_ready"
    ALL_ROOTS_TOO_CLOSE_TO_EDGE = "diagnostic_all_roots_too_close_to_edge"
    NAN_VALUES_DETECTED = "diagnostic_nan_values_detected"
    INF_VALUES_DETECTED = "diagnostic_inf_values_detected"


class SnuffledDiagnostics(NamedArray):
    """Object providing detected (snuffled) values for all Diagnostic members."""

    def __init__(self, values: list[float] | None = None) -> None:
        super().__init__(names=list(Diagnostic), values=values)
