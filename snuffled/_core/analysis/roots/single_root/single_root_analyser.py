from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models import RootProperty, SnuffledRootProperties
from snuffled._core.utils.constants import SEED_OFFSET_SINGLE_ROOT_ANALYSER


class SingleRootAnalyser(PropertyExtractor[SnuffledRootProperties]):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        function_sampler: FunctionSampler,
        root: tuple[float, float],
        dx: float,
        n_root_samples: int,
        seed: int = 42,
    ):
        super().__init__(function_sampler)
        self.root = root
        self.dx = dx
        self.n_root_samples = n_root_samples
        self._seed = seed + SEED_OFFSET_SINGLE_ROOT_ANALYSER

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
            case RootProperty.ASYMMETRIC:
                return self._extract_asymmetric()
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

    def _extract_asymmetric(self) -> float:
        return -1.0
