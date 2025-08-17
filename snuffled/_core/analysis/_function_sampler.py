import math
from collections.abc import Callable
from functools import cache
from typing import Literal, LiteralString

import numpy as np

from snuffled._core.compatibility import numba
from snuffled._core.utils.sampling import multi_scale_samples
from tests.utils.constants import EPS


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

        # pre-fill cache, which is faster than going through self.f(x) for each x
        fx_values = np.zeros_like(x_values, dtype=float)
        for i, x in enumerate(x_values):
            if x not in self._fun_cache:
                fx = self._fun(x)
                self._fun_cache[x] = self._fun(x)
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
    def smooth_fx_sign(self) -> np.ndarray:
        """
        Returns an array with elements in [-1,+1] representing a more nuance np.sign(fx_values()).

        Local and global tolerance areas are used to determine the threshold around which differences transition
        (smoothly) from 0 to 1 or 0 to -1.

        See also smooth_sign().
        """

        # determine inner_tol, outer_tol
        tol_global = self.tol_array_global()
        tol_local = self.tol_array_local()

        inner_tol = np.minimum(tol_local, tol_global)  # make sure inner_tol[i] is the smallest of both tolerances
        outer_tol = np.maximum(tol_local, tol_global)  # make sure outer_tol[i] is the largest of both tolerances

        # reduce to size n-1 from size n  (take maximum
        inner_tol = np.maximum(inner_tol[1:], inner_tol[:-1])  # take maximum of all subsequent points
        outer_tol = np.maximum(outer_tol[1:], outer_tol[:-1])  # take maximum of all subsequent points

        # compute smooth_sign
        return smooth_sign(
            values=np.diff(self.fx_values()),
            inner_tol=inner_tol,
            outer_tol=outer_tol,
        )


# =================================================================================================
#  Static Helpers
# =================================================================================================
@numba.njit
def smooth_sign(values: np.ndarray, inner_tol: np.ndarray, outer_tol: np.ndarray) -> np.ndarray:
    """
    Returns an array that can be seen as a 'smooth' version of np.sign(values).

    Smoothing should NOT be interpreted in the i-direction along the array, but rather in the value-direction,
    i.e. we replace the normally expected values -1, 0, +1 by the entire spectrum of values in [-1, 1].
    This is done by taking into account the 'inner_tol' and 'outer_tol' arrays.

    The following mapping is used for mapping f[i+1]-f[i] to [-1, 1]:

        f[i+1]-f[i]                 smooth_fx_sign

           >> +outer_tol    -->         +1.00
              +outer_tol    -->      ~  +0.75
              +inner_tol    -->      ~  +0.25
               0.0          -->          0.00
              -inner_tol    -->      ~  -0.25
              -outer_tol    -->      ~  -0.75
           << -outer_tol    -->         -1.00

    This implementation assumes that 0 <= inner_tol <= outer_tol.
    """

    def sigmoid_like(x: float) -> float:
        """Linear around x=0, levels off at -1 and +1 for ±inf; f(1)=1/2"""
        return x / math.sqrt(3 + (x * x))

    result = np.zeros_like(values)
    for i, (v, inner, outer) in enumerate(zip(values, inner_tol, outer_tol)):
        if v == 0.0:
            result[i] = 0.0
        elif inner > 0.0:
            result[i] = 0.5 * (sigmoid_like(v / inner) + sigmoid_like(v / outer))
        elif outer > 0.0:
            result[i] = 0.5 * (1.0 + sigmoid_like(v / outer))
        else:
            result[i] = 1.0

    return result


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
