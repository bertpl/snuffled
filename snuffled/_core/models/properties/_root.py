from snuffled._core.compatibility import StrEnum
from snuffled._core.models.base import NamedArray


class RootProperty(StrEnum):
    ILL_BEHAVED = "root_ill_behaved"
    DERIVATIVE_ZERO = "root_derivative_zero"
    DERIVATIVE_INFINITE = "root_derivative_infinite"
    DISCONTINUOUS = "root_discontinuous"
    ASYMMETRIC = "root_asymmetric"


class SnuffledRootProperties(NamedArray):
    """
    Object providing detected (snuffled) values for all RootProperty members.
    """

    def __init__(self, values: list[float] | None = None):
        super().__init__(names=list(RootProperty), values=values)
