from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.models import RootProperty, SnuffledRootProperties


class RootsAnalyser:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_data: FunctionSampler, dx: float, n_root_samples):
        self.function_data = function_data
        self.dx = dx
        self.n_root_samples = n_root_samples

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def analyse(self) -> SnuffledRootProperties:
        props = SnuffledRootProperties()
        props[RootProperty.DERIVATIVE_ZERO] = self._detect_derivative_zero()
        props[RootProperty.DERIVATIVE_INFINITE] = self._detect_derivative_infinite()
        props[RootProperty.NOISY] = self._detect_noisy()
        props[RootProperty.DISCONTINUOUS] = self._detect_discontinuous()
        props[RootProperty.NON_DIFFERENTIABLE] = self._detect_non_differentiable()
        return props

    # -------------------------------------------------------------------------
    #  Internal snuffling methods
    # -------------------------------------------------------------------------
    def _detect_derivative_zero(self) -> float:
        return -1.0

    def _detect_derivative_infinite(self) -> float:
        return -1.0

    def _detect_noisy(self) -> float:
        return -1.0

    def _detect_discontinuous(self) -> float:
        return -1.0

    def _detect_non_differentiable(self) -> float:
        return -1.0
