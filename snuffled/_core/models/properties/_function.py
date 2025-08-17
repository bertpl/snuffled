from snuffled._core.compatibility import StrEnum
from snuffled._core.models.base import NamedArray


class FunctionProperty(StrEnum):
    MANY_ZEROES = "function_many_zeroes"
    NON_MONOTONIC = "function_non_monotonic"
    HIGH_DYNAMIC_RANGE = "function_high_dynamic_range"
    DISCONTINUOUS = "function_discontinuous"
    FLAT_INTERVALS = "function_flat_intervals"


class SnuffledFunctionProperties(NamedArray):
    """
    Object providing detected (snuffled) values for all FunctionProperty members.
    """

    def __init__(self, values: list[float] | None = None):
        super().__init__(names=list(FunctionProperty), values=values)
