from snuffled._core.compatibility import StrEnum
from snuffled._core.models.base import NamedArray


class Diagnostic(StrEnum):
    MAX_ZERO_WIDTH = "diagnostic_max_zero_width"
    SIGN_CHANGE_MISSING = "diagnostic_sign_change_missing"


class SnuffledDiagnostics(NamedArray):
    """
    Object providing detected (snuffled) values for all Diagnostic members.
    """

    def __init__(self, values: list[float] | None = None):
        super().__init__(names=list(Diagnostic), values=values)
