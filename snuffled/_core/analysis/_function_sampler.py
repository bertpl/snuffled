import math
from collections.abc import Callable
from functools import cache
from typing import overload

import numpy as np

from snuffled._core.models.root_analysis import Root
from snuffled._core.utils.constants import EPS, FX_CLIP, SEED_OFFSET_FUNCTION_SAMPLER
from snuffled._core.utils.math import smooth_sign_array
from snuffled._core.utils.root_finding import find_odd_root
from snuffled._core.utils.sampling import max_x_delta, multi_scale_samples, sample_integers


class FunctionSampler:
    """Class holding (cached) data related to a specific function, shared across different analyses.

    All the following classes make use of this class to access the data they need for their analyses:
      - FunctionAnalyser
      - RootsAnalyser
      - DiagnosticAnalyser.
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
        n_fun_samples: int = 1_000,
        n_roots: int = 100,
        rel_tol_scale: float = 10.0,
    ) -> None:
        # --- randomization -------------------------------
        self._seed = seed + SEED_OFFSET_FUNCTION_SAMPLER

        # --- function properties -------------------------
        self._fun = fun
        self.x_min = x_min
        self.x_max = x_max

        # --- settings ------------------------------------
        self.dx = dx
        self.n_fun_samples = n_fun_samples
        self.n_roots = n_roots
        self.rel_tol = EPS * rel_tol_scale

        # --- cache ---------------------------------------
        self._fun_cache: dict[float, float] = {}

        # --- non-finite tracking -------------------------
        # set in _eval() when the function returns NaN / +-inf for a sampled point; read by the
        # NAN_VALUES_DETECTED / INF_VALUES_DETECTED diagnostics.
        self._saw_nan = False
        self._saw_inf = False

    # -------------------------------------------------------------------------
    #  Low-level generic functionality
    # -------------------------------------------------------------------------
    @overload
    def f(self, x: float) -> float: ...
    @overload
    def f(self, x: list[float]) -> list[float]: ...
    def f(self, x: float | list[float]) -> float | list[float]:
        # simple cached version of fun(x), without size limit (and simpler than lru_cache, so less overhead)
        if isinstance(x, list):
            # --- get MULTIPLE f(x) values ------
            fx_values = []
            for single_x in x:
                if not (self.x_min <= single_x <= self.x_max):
                    raise ValueError(f"x={single_x} is out of bounds [{self.x_min}, {self.x_max}]")
                if single_x not in self._fun_cache:
                    self._fun_cache[single_x] = fx = self._eval(single_x)
                else:
                    fx = self._fun_cache[single_x]
                fx_values.append(fx)
            return fx_values
        # --- get SINGLE f(x) values --------
        if x in self._fun_cache:
            return self._fun_cache[x]
        if self.x_min <= x <= self.x_max:
            fx = self._eval(x)
            self._fun_cache[x] = fx
            return fx
        raise ValueError(f"x={x} is out of bounds [{self.x_min}, {self.x_max}]")

    def _eval(self, x: float) -> float:
        """Evaluate fun(x), sanitizing non-finite results so no NaN / +-inf reaches the analysis kernels.

        NaN -> FX_CLIP (and _saw_nan set: per the NAN_VALUES_DETECTED contract, all other metrics are
        then unspecified). We use FX_CLIP rather than 0.0 only to avoid introducing a fake zero-region
        that would re-trigger the high-dynamic-range divide-by-zero; the exact value is otherwise
        immaterial. +-inf -> sign * FX_CLIP (and _saw_inf set); finite magnitudes above FX_CLIP are
        clipped too (arithmetic safety), without setting a flag.
        """
        raw = float(self._fun(x))
        if math.isnan(raw):
            self._saw_nan = True
            return FX_CLIP
        if math.isinf(raw):
            self._saw_inf = True
            return math.copysign(FX_CLIP, raw)
        if abs(raw) > FX_CLIP:
            return math.copysign(FX_CLIP, raw)
        return raw

    @property
    def saw_nan(self) -> bool:
        """Whether the function returned NaN for at least one sampled point so far."""
        return self._saw_nan

    @property
    def saw_inf(self) -> bool:
        """Whether the function returned +-inf for at least one sampled point so far."""
        return self._saw_inf

    @cache
    def x_values(self) -> np.ndarray:
        """Returns an array of x-values in [x_min, x_max] that we use to sample the function and infer its properties.

        This is based on the multi_scale_samples function and does not include any other cached x-values resulting
        from other calls to .f(.).
        """
        return multi_scale_samples(
            x_min=self.x_min,
            x_max=self.x_max,
            dx_min=self.dx,
            n=self.n_fun_samples,
            seed=self._seed,
        )

    @cache
    def fx_values(self) -> np.ndarray:
        """f(x) values corresponding to the x_values()."""
        return np.array(self.f(list(self.x_values())), dtype=np.float64)

    def function_cache(self) -> list[tuple[float, float]]:
        """Returns contents of the function cache as a list of (x, f(x))-tuples.

        Note this might return more information than .x_values() and .fx_values(), since those methods
        only return information related to the initial multiscale sampling.
        """
        return list(self._fun_cache.items())

    def function_cache_size(self) -> int:
        """Return number of values in function cache."""
        return len(self._fun_cache)

    @cache
    def fx_diff_values(self) -> np.ndarray:
        return np.diff(self.fx_values())

    @cache
    def fx_quantile(self, q: float, absolute: bool) -> float:
        """Returns the requested quantile f(x).

        absolute==False    --> quantile of f(x)
        absolute==True     --> quantile of abs(f(x)).
        """
        if absolute:
            return float(np.quantile(abs(self.fx_values()), q))
        return float(np.quantile(self.fx_values(), q))

    @cache
    def robust_estimated_fx_max(self) -> float:
        """Robust, approximate estimate of max(f(x)), not susceptible to single-sample outliers.

        Such outliers might arise in certain corner cases.

        NOTE: this value is not guaranteed to be equal or larger than abs(f(x)), but should provide a reasonable
              estimate under most regular circumstances.
        """
        q = 1 - (1 / math.sqrt(self.n_fun_samples))
        return (1 / q) * self.fx_quantile(q, absolute=True)

    # -------------------------------------------------------------------------
    #  Specialized - Tolerances
    # -------------------------------------------------------------------------
    @cache
    def tol_array_local(self) -> np.ndarray:
        """Return (n_fun_samples,)-sized array with absolute (>0) LOCAL tolerance values.

        These represent the tolerance wrt numerical rounding errors on a LOCAL per-sample basis, i.e. computed
        based on the magnitude of each f(x) sample.
        """
        return self.rel_tol * abs(self.fx_values())

    @cache
    def tol_array_global(self) -> np.ndarray:
        """Return (n_fun_samples,)-sized array with absolute (>0) GLOBAL tolerance values.

        These represent the tolerance wrt numerical rounding errors on a GLOBAL basis, i.e. computed based on the
        overall (~maximum) magnitude of f(x).  This is a constant array.
        """
        return np.full(self.n_fun_samples, self.rel_tol * self.robust_estimated_fx_max())

    @cache
    def fx_diff_smooth_sign(self) -> np.ndarray:
        """Returns an array with elements in [-1,+1] representing a more nuanced np.sign(np.diff(fx_values())).

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

    # -------------------------------------------------------------------------
    #  Specialized - Roots
    # -------------------------------------------------------------------------
    @cache
    def candidate_root_intervals(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return a list of 'root intervals' and 'non-root intervals' for root finding.

        The root intervals serve as candidates (but potentially too many) for root finding.

        :return: (root_intervals, non_root_intervals)-tuple
                    root_intervals      : list of (x_left, x_right)-tuples   WITH    sign flip
                    non_root_intervals  : list of (x_left, x_right)-tuples   WITHOUT sign flip.
        """
        # multi-scale samples as (x, fx)-tuples
        samples = list(zip(self.x_values(), self.fx_values()))

        # remove 0-valued samples, we are looking for strict sign flips
        samples = [(x, fx) for i, (x, fx) in enumerate(samples) if (fx != 0.0)]

        # split in root- and non-root-intervals
        root_intervals = []
        non_root_intervals = []
        for (x_left, fx_left), (x_right, fx_right) in zip(samples[:-1], samples[1:]):
            if np.sign(fx_left) != np.sign(fx_right):
                # sign flip
                root_intervals.append((x_left, x_right))
            else:
                # no sign flip
                non_root_intervals.append((x_left, x_right))

        # we're done
        return root_intervals, non_root_intervals

    @cache
    def roots(self) -> list[Root]:
        """Returns at most 'n_roots' root intervals [root_min, root_max] obtained using find_root_and_width().

        We start from the candidate_root_intervals and - if needed - sample 'n_roots' intervals randomly
        to if there are too many candidate intervals.
        :return: list of Root objects, with deriv_sign != 0.
        """
        # get intervals
        cand_intervals, _ = self.candidate_root_intervals()

        # sample if needed
        if len(cand_intervals) > self.n_roots:
            cand_intervals = [
                cand_intervals[i]
                for i in sample_integers(
                    i_min=0,
                    i_max=len(cand_intervals),
                    n=self.n_roots,
                    seed=self._seed,
                )
            ]

        # compute roots & return
        return [
            find_odd_root(
                fun=self.f,
                x_min=x_min,
                x_max=x_max,
                dx_min=(EPS * EPS) * (self.x_max - self.x_min),
            )
            for x_min, x_max in cand_intervals
        ]

    @cache
    def analyzable_roots(self) -> list[Root]:
        """Roots far enough from the interval edges for reliable two-sided analysis.

        A root is analyzable only if both sides have room for the full sampling span
        `max_x_delta(dx)`; nearer the edge, sampling `root.x ± x_delta` would fall outside
        `[x_min, x_max]`. Scoped to two-sided root analysis — `roots()` still returns every root
        (for counting, zero-width, and the all-too-close diagnostic).
        """
        margin = max_x_delta(self.dx)
        return [root for root in self.roots() if (root.x - self.x_min >= margin) and (self.x_max - root.x >= margin)]
