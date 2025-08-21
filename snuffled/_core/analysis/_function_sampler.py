import math
from collections.abc import Callable
from functools import cache
from typing import Literal

import numpy as np

from snuffled._core.compatibility import numba
from snuffled._core.utils.constants import EPS
from snuffled._core.utils.math import smooth_sign_array
from snuffled._core.utils.sampling import multi_scale_samples


class FunctionSampler:
    """
    Class holding (cached) data related to a specific function, shared across different analyses.
    All the following classes make use of this class to access the data they need for their analyses:
     - FunctionAnalyser
     - RootAnalyser
     - DiagnosticAnalyser
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        fun: Callable[[float], float],
        x_min: float,
        x_max: float,
        dx: float = 1e-8,
        n_fun_samples: int = 1_000,
        rel_tol_scale: float = 10.0,
    ):
        # --- function properties -------------------------
        self._fun = fun
        self.x_min = x_min
        self.x_max = x_max

        # --- settings ------------------------------------
        self.dx = dx
        self.n_fun_samples = n_fun_samples
        self.rel_tol = EPS * rel_tol_scale

        # --- cache ---------------------------------------
        self._fun_cache: dict[float, float] = dict()

    # -------------------------------------------------------------------------
    #  Low-level generic functionality
    # -------------------------------------------------------------------------
    def f(self, x: float) -> float:
        # simple cached version of fun(x), without size limited (and simpler than lru_cache, so less overhead)
        if x in self._fun_cache:
            return self._fun_cache[x]
        elif self.x_min <= x <= self.x_max:
            fx = self._fun(x)
            self._fun_cache[x] = fx
            return fx
        else:
            raise ValueError(f"x={x} is out of bounds [{self.x_min}, {self.x_max}]")

    @cache
    def x_values(self) -> np.ndarray:
        """
        Returns an array of x-values in [x_min, x_max] that we use to sample the function and infer its properties.

        This is based on the multi_scale_samples function.
        """
        return multi_scale_samples(
            x_min=self.x_min,
            x_max=self.x_max,
            dx_min=self.dx,
            n=self.n_fun_samples,
        )

    @cache
    def fx_values(self, smoothing: Literal["", "absolute", "relative"] = "") -> np.ndarray:
        """
        f(x) values corresponding to the x_values().
        """

        # get x-values
        x_values = self.x_values()

        # use _fun(.) and fill cache, which is faster than going through self.f(x) for each x
        fx_values = np.zeros_like(x_values, dtype=float)
        for i, x in enumerate(x_values):
            if x not in self._fun_cache:
                self._fun_cache[x] = fx = self._fun(x)
            else:
                fx = self._fun_cache[x]
            fx_values[i] = fx

        # check if we need to smoothen
        if smoothing == "":
            return fx_values
        elif smoothing == "absolute":
            return smoothen_fx_abs_tol(fx=fx_values, abs_tol=self.rel_tol * self.fx_quantile(0.9, absolute=True))
        elif smoothing == "relative":
            return smoothen_fx_rel_tol(fx=fx_values, rel_tol=self.rel_tol)
        else:
            raise ValueError(f"unsupported value for smoothing parameter: {smoothing}")

    def function_cache(self) -> list[tuple[float, float]]:
        """
        Returns contents of the function cache as a list of (x, f(x))-tuples.
        Note this might return more information than .x_values() and .fx_values(), since those methods
        only return information related to the initial multiscale sampling.
        """
        return list(self._fun_cache.items())

    @cache
    def fx_diff_values(self) -> np.ndarray:
        return np.diff(self.fx_values())

    @cache
    def fx_quantile(self, q: float, absolute: bool) -> float:
        """
        Returns the requested quantile f(x).
            absolute==False    --> quantile of f(x)
            absolute==True     --> quantile of abs(f(x))
        """
        if absolute:
            return float(np.quantile(abs(self.fx_values()), q))
        else:
            return float(np.quantile(self.fx_values(), q))

    @cache
    def robust_estimated_fx_max(self) -> float:
        """
        Robust, approximate estimate of max(f(x)), without being susceptible to single-sample outliers, which might
        arise in certain corner cases.

        NOTE: this value is not guaranteed to be equal or larger than abs(f(x)), but should provide a reasonable
              estimate under most regular circumstances.
        """
        q = 1 - (1 / math.sqrt(self.n_fun_samples))
        return (1 / q) * self.fx_quantile(q, absolute=True)

    # -------------------------------------------------------------------------
    #  Specialized
    # -------------------------------------------------------------------------
    @cache
    def tol_array_local(self) -> np.ndarray:
        """
        Return (n_fun_samples, )-sized array with absolute tolerance values > 0, representing the LOCAL tolerance wrt
        numerical rounding errors on a LOCAL per-sample basis, i.e. computed based on the magnitude of each f(x) sample.
        """
        return self.rel_tol * abs(self.fx_values())

    @cache
    def tol_array_global(self) -> np.ndarray:
        """
        Return (n_fun_samples, )-sized array with absolute tolerance values > 0, representing the GLOBAL tolerance wrt
        numerical rounding errors on a GLOBAL basis, i.e. computed based on the overall (~maximum) magnitude of f(x).
        This is a constant matrix.
        """
        return np.full(self.n_fun_samples, self.rel_tol * self.robust_estimated_fx_max())

    @cache
    def fx_diff_smooth_sign(self) -> np.ndarray:
        """
        Returns an array with elements in [-1,+1] representing a more nuanced np.sign(np.diff(fx_values())).

        Local and global tolerances are used to determine the threshold around which differences transition
        (smoothly) from 0 to 1 or 0 to -1.

        See also smooth_sign().
        """

        # determine inner_tol, outer_tol
        tol_global = self.tol_array_global()
        tol_local = self.tol_array_local()

        inner_tol = np.minimum(tol_local, tol_global)  # make sure inner_tol[i] is the smallest of both tolerances
        outer_tol = np.maximum(tol_local, tol_global)  # make sure outer_tol[i] is the largest of both tolerances

        # reduce to size n-1 from size n  (take sum)
        inner_tol = inner_tol[1:] + inner_tol[:-1]  # take sum of both tolerances
        outer_tol = outer_tol[1:] + outer_tol[:-1]  # take sum of both tolerances

        # compute smooth_sign
        return smooth_sign_array(
            x=self.fx_diff_values(),
            inner_tol=inner_tol,
            outer_tol=outer_tol,
        )


# =================================================================================================
#  Static Helpers
# =================================================================================================
@numba.njit
def smoothen_fx_abs_tol(fx: np.ndarray, abs_tol: float) -> np.ndarray:
    """
    Smoothen a series of f(x) values (for increasing x) such that we keep f(x) values constant
    as long as they stay within a certain bound of the starting f(x') value of the current run of constant values.

    An ABSOLUTE tolerance is used, i.e. we check if abs(f(x) - f(x')) <= abs_tol
    """
    fx_smooth = fx.copy()
    fx_ref = fx[0]
    for i in range(1, len(fx)):
        if abs(fx_ref - fx[i]) <= abs_tol:
            # keep constant
            fx_smooth[i] = fx_ref
        else:
            # deviation large enough, don't keep constant
            fx_ref = fx[i]
    return fx_smooth


@numba.njit
def smoothen_fx_rel_tol(fx: np.ndarray, rel_tol: float) -> np.ndarray:
    """
    Smoothen a series of f(x) values (for increasing x) such that we keep f(x) values constant
    as long as they stay within a certain bound of the starting f(x') value of the current run of constant values.

    A RELATIVE tolerance is used, i.e. we check if abs(f(x) - f(x')) <= rel_tol * abs(f(x))
    """
    fx_smooth = fx.copy()
    fx_ref = fx[0]
    for i in range(1, len(fx)):
        if abs(fx_ref - fx[i]) <= (rel_tol * abs(fx[i])):
            # keep constant
            fx_smooth[i] = fx_ref
        else:
            # deviation large enough, don't keep constant
            fx_ref = fx[i]
    return fx_smooth
