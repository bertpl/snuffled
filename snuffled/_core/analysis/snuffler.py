from functools import lru_cache
from typing import Callable

from snuffled._core.models import SnuffledFunctionProperties, SnuffledProperties, SnuffledRootProperties
from snuffled._core.models.properties import SnuffledDiagnostics

from ._function_sampler import FunctionSampler
from .diagnostic import DiagnosticAnalyser
from .function import FunctionAnalyser
from .roots import RootsAnalyser


class Snuffler:
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
        n_root_samples: int = 100,
        n_fun_samples: int = 10_000,
        rel_tol: float = 10.0,
        x_jitter: float = 1.0,
    ):
        self._function_data = FunctionSampler(fun, x_min, x_max, n_fun_samples, rel_tol, x_jitter)
        self._function_analyser = FunctionAnalyser(self._function_data, rel_tol, x_jitter)
        self._roots_analyser = RootsAnalyser(self._function_data, dx, n_root_samples)
        self._diagnostics_analyser = DiagnosticAnalyser(self._function_data)

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def all(self) -> SnuffledProperties:
        return SnuffledProperties(
            function_props=self.function(),
            root_props=self.roots(),
            diagnostics=self.diagnostics(),
        )

    @lru_cache(maxsize=1)
    def function(self) -> SnuffledFunctionProperties:
        return self._function_analyser.analyse()

    @lru_cache(maxsize=1)
    def roots(self) -> SnuffledRootProperties:
        return self._roots_analyser.analyse()

    @lru_cache(maxsize=1)
    def diagnostics(self) -> SnuffledDiagnostics:
        return self._diagnostics_analyser.analyse()
