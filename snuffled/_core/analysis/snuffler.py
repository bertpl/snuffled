from collections.abc import Callable

from snuffled._core.models import (
    Diagnostic,
    FunctionProperty,
    RootProperty,
    SnuffledProperties,
)
from snuffled._core.utils.constants import (
    DEFAULT_N_FUN_SAMPLES,
    DEFAULT_N_ROOT_SAMPLES,
    DEFAULT_N_ROOTS,
    SEED_OFFSET_SNUFFLER,
)

from ._function_sampler import FunctionSampler
from ._property_extractor import PropertyExtractor
from .diagnostic import DiagnosticAnalyser
from .function import FunctionAnalyser
from .roots import RootsAnalyser


class Snuffler(PropertyExtractor[SnuffledProperties]):
    """Analyze a function, returning SnuffledRootProperties, SnuffledFunctionProperties, or all SnuffledProperties.

    Non-finite function values are handled at the sampling boundary so analysis never crashes:
      - ``+-inf`` is clipped to ``+-FX_CLIP`` and still characterized (a pole reads as discontinuous);
        ``INF_VALUES_DETECTED`` flags that it occurred.
      - ``NaN`` sets ``NAN_VALUES_DETECTED`` and, per contract, leaves every *other* metric unspecified
        (finite, but its value may be anything) — a NaN-returning function is not a well-defined target.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        fun: Callable[[float], float],
        x_min: float,
        x_max: float,
        dx: float,
        seed: int,
        n_fun_samples: int = DEFAULT_N_FUN_SAMPLES,
        n_roots: int = DEFAULT_N_ROOTS,
        n_root_samples: int = DEFAULT_N_ROOT_SAMPLES,
        rel_tol_scale: float = 10.0,
    ) -> None:
        seed += SEED_OFFSET_SNUFFLER
        function_sampler = FunctionSampler(fun, x_min, x_max, dx, seed, n_fun_samples, n_roots, rel_tol_scale)
        super().__init__(function_sampler)
        self._function_analyser = FunctionAnalyser(function_sampler)
        self._roots_analyser = RootsAnalyser(function_sampler, n_root_samples, seed)
        self._diagnostics_analyser = DiagnosticAnalyser(function_sampler)

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def _new_named_array(self) -> SnuffledProperties:
        return SnuffledProperties()

    def supported_properties(self) -> list[str]:
        """Order the extraction so shared sampling is done before it's needed.

        Root then function analysis run first (function still benefits from the roots' samples); the
        diagnostics come last so the NAN_VALUES_DETECTED / INF_VALUES_DETECTED flags reflect every
        f(x) the deeper analyses evaluated (root bisection, discontinuity zoom), not just the initial
        multi-scale grid.
        """
        roots_props = self._roots_analyser.supported_properties()
        function_props = self._function_analyser.supported_properties()
        diagnostic_props = self._diagnostics_analyser.supported_properties()

        return roots_props + function_props + diagnostic_props

    def _extract(self, prop: str) -> float:
        if isinstance(prop, Diagnostic):
            return self._diagnostics_analyser.extract(prop)
        if isinstance(prop, RootProperty):
            return self._roots_analyser.extract(prop)
        if isinstance(prop, FunctionProperty):
            return self._function_analyser.extract(prop)
        raise ValueError(f"Property {prop} not supported.")
