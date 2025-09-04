from typing import Callable

from snuffled._core.models import (
    Diagnostic,
    FunctionProperty,
    RootProperty,
    SnuffledProperties,
)
from snuffled._core.utils.constants import SEED_OFFSET_SNUFFLER

from ._function_sampler import FunctionSampler
from ._property_extractor import PropertyExtractor
from .diagnostic import DiagnosticAnalyser
from .function import FunctionAnalyser
from .roots import RootsAnalyser


class Snuffler(PropertyExtractor[SnuffledProperties]):
    """
    Class for analyzing a function, returning either SnuffledRootProperties,
    SnuffledFunctionProperties, or all SnuffledProperties.
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
        n_fun_samples: int = 10_000,
        n_roots: int = 100,
        n_root_samples: int = 100,
        rel_tol_scale: float = 10.0,
    ):
        seed += SEED_OFFSET_SNUFFLER
        function_sampler = FunctionSampler(fun, x_min, x_max, dx, seed, n_fun_samples, n_roots, rel_tol_scale)
        super().__init__(function_sampler)
        self._function_analyser = FunctionAnalyser(function_sampler)
        self._roots_analyser = RootsAnalyser(function_sampler, n_root_samples, seed)
        self._diagnostics_analyser = DiagnosticAnalyser(function_sampler)

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def supported_properties(self) -> list[str]:
        """Make sure function analysis is performed last, so it benefits from samples taken by other analyses."""
        diagnostic_props = self._diagnostics_analyser.supported_properties()
        roots_props = self._roots_analyser.supported_properties()
        function_props = self._function_analyser.supported_properties()

        return diagnostic_props + roots_props + function_props

    def _extract(self, prop: str) -> float:
        if isinstance(prop, Diagnostic):
            return self._diagnostics_analyser.extract(prop)
        elif isinstance(prop, RootProperty):
            return self._roots_analyser.extract(prop)
        elif isinstance(prop, FunctionProperty):
            return self._function_analyser.extract(prop)
        else:
            raise ValueError(f"Property {prop} not supported.")
