from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models import RootProperty, SnuffledRootProperties


class RootsAnalyser(PropertyExtractor[SnuffledRootProperties]):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_sampler: FunctionSampler, dx: float, n_root_samples: int):
        super().__init__(function_sampler)
        self.dx = dx
        self.n_root_samples = n_root_samples

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def _new_named_array(self) -> SnuffledRootProperties:
        return SnuffledRootProperties()

    def _extract(self, prop: str) -> float:
        match prop:
            case RootProperty.DISCONTINUOUS:
                return self._extract_discontinuous()
            case RootProperty.DERIVATIVE_ZERO:
                return self._extract_derivative_zero()
            case RootProperty.DERIVATIVE_INFINITE:
                return self._extract_derivative_infinite()
            case RootProperty.NOISY:
                return self._extract_noisy()
            case RootProperty.NON_DIFFERENTIABLE:
                return self._extract_non_differentiable()
            case _:
                raise ValueError(f"Property {prop} not supported")

    # -------------------------------------------------------------------------
    #  Internal methods
    # -------------------------------------------------------------------------
    def _extract_derivative_zero(self) -> float:
        return -1.0

    def _extract_derivative_infinite(self) -> float:
        return -1.0

    def _extract_noisy(self) -> float:
        return -1.0

    def _extract_discontinuous(self) -> float:
        return -1.0

    def _extract_non_differentiable(self) -> float:
        return -1.0
