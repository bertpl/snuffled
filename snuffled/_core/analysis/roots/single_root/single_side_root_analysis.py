import math
from functools import cached_property

import numpy as np

from snuffled._core.analysis.roots.single_root.curve_fitting import fitting_curve
from snuffled._core.analysis.roots.single_root.curve_fitting._fit_tailored import fit_curve_with_uncertainty_tailored
from snuffled._core.utils.constants import EPS
from snuffled._core.utils.numba import clip_scalar

# =================================================================================================
#  Constants
# =================================================================================================
_A_RANGE_MIN = EPS**0.25  # such that A_RANGE_MAX / A_RANGE_MIN is still well within bounds where
_A_RANGE_MAX = 1 / _A_RANGE_MIN  # numerical accuracy of such ratios starts to break down

_B_RANGE_MIN = -0.5  # this should allow the range of b-values to encompass 0.0 in case of no discontinuity
_B_RANGE_MAX = 1.0  # this will max out the score for discontinuity

_C_RANGE_MIN = 1 / 8.0  # smaller c-values will have g(1) > ~0.9*g(2), so c-estimation can become ill-conditioned
_C_RANGE_MAX = 16.0  # larger c-values will get close to causing underflow for EPS^c


# =================================================================================================
#  Single-Side Root Analysis
# =================================================================================================
class SingleSideRootAnalysis:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, dx: float, x_deltas: np.ndarray, fx_values: np.ndarray, dx_sign: int, fx_sign: int):
        """

        :param dx: (float > 0) dx-value used to generate x_deltas
        :param x_deltas: (n,)-sized numpy array; x_deltas > 0 as generated using compute_x_deltas()
        :param fx_values: (n,)-sized numpy array; function values f(root+(dx_sign*x_deltas))
        :param dx_sign: (int; +1 or -1) -1 if this is a left-sided analysis, +1 if this is a right-sided analysis
        :param fx_sign: (int; +1 or -1) expected dominant sign of fx_values on this side of the root.
                                        We'll multiply the fx-values with this sign before we apply curve fitting.
        """

        # --- store data ----------------------------------
        self.dx = dx
        self.x_deltas = x_deltas
        self.fx_values = fx_values
        self.dx_sign = dx_sign
        self.fx_sign = fx_sign

        # --- pre-processed data --------------------------
        (x, fx), (x_scale, fx_scale) = self.preprocess_x_fx()
        self.x_pre = x
        self.fx_pre = fx
        self.x_scale = x_scale
        self.fx_scale = fx_scale

        # --- analysis ------------------------------------
        # TODO:
        #   - distinguish 3 cases:
        #       1) all fx_values are exactly 0.0
        #       2) regular curve fitting with positive c-values
        #       3) inverse curve fitting with negative c-values
        #   - validate curve_fitting can handle negative-valued range_c argument
        a_values, b_values, c_values, cost_values = fit_curve_with_uncertainty_tailored(
            x=x,
            fx=fx,
            range_a=(_A_RANGE_MIN, _A_RANGE_MAX),
            range_b=(_B_RANGE_MIN, _B_RANGE_MAX),
            range_c=(_C_RANGE_MIN, _C_RANGE_MAX),
            reg=1e-3,
            n_iters=15,
            rel_uncertainty_size=1.0,
        )
        self._a_values = a_values
        self._b_values = b_values
        self._c_values = c_values
        self._cost_values = cost_values

    # -------------------------------------------------------------------------
    #  Internal analysis methods
    # -------------------------------------------------------------------------
    def preprocess_x_fx(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float]]:
        """
        Returns (x, fx), (x_scale, fx_scale)-tuples, such that (x, fx) can be used directly in curve fitting methods.

        Roughly the following processing happens
            x  = x_deltas / x_scale                      (such that median(x)==1.0)
            fx = fx_orig / (fx_sign * fx_scale)

        The goal is to avoid ill-conditioning during curve fitting and have scaling such that we expect
        roughly ~1 values of parameter a, such that it is safe to impose bounds (_A_RANGE_MIN, _A_RANGE_MAX) on a.

        fx_scale, in most cases, will coincide with median(fx / x), but we take precautions for the case there are
        some (or many) fx-values <= 0, to robustly deal with such corner/degenerate cases.
        """

        # compute x_scale & x
        x_scale = 2 * self.dx  # should coincide with median(x_deltas)
        x = self.x_deltas / x_scale

        # compute fx_scale
        # NOTE: Ideally we want to choose the fx_scale, knowing that we will use range_a=(_A_RANGE_MIN, _A_RANGE_MAX).
        #       So we will try to choose fx_scale such that for c=1.0, values of a ranging in 'range_a' can go
        #       through most of the data points.
        fx_pos = self.fx_sign * self.fx_values  # these should (mostly) be positive values
        fx_x_ratios = [abs(fx / x) for fx, x in zip(fx_pos, x) if fx != 0]  # since not all fx==0, this is not empty
        fx_scale_max = max(fx_x_ratios)
        fx_scale = clip_scalar(
            float(np.median(fx_x_ratios)),  # guaranteed to be a strictly positive value
            fx_scale_max / _A_RANGE_MAX,
            fx_scale_max,
        )

        # compute fx
        fx = self.fx_values / (self.fx_sign * fx_scale)

        # return
        return (x, fx), (x_scale, fx_scale)

    # -------------------------------------------------------------------------
    #  Parameter aggregation
    # -------------------------------------------------------------------------
    @cached_property
    def _a_min_max(self) -> tuple[float, float]:
        a_min = float(np.min(self._a_values))
        a_max = float(np.max(self._a_values))
        return a_min, a_max

    @cached_property
    def _b_min_max(self) -> tuple[float, float]:
        b_min = float(np.min(self._b_values))
        b_max = float(np.max(self._b_values))
        return b_min, b_max

    @cached_property
    def _c_min_max(self) -> tuple[float, float]:
        c_min = float(np.min(self._c_values))
        c_max = float(np.max(self._c_values))
        return c_min, c_max

    @cached_property
    def _abc_tuples(self) -> list[tuple[float, float, float]]:
        return list(zip(self._a_values, self._b_values, self._c_values))

    @cached_property
    def _abc_opt(self) -> tuple[float, float, float]:
        i_opt = np.argmin(self._cost_values)
        return (
            self._a_values[i_opt],
            self._b_values[i_opt],
            self._c_values[i_opt],
        )

    def _f_min_max(self, z: float) -> tuple[float, float]:
        """Return (min,max)-values for estimated f(root + z*2*dx) based on (a,b,c)-params in uncertainty range."""
        if z == 0.0:
            g_values = [a * b for a, b, c in self._abc_tuples]
        else:
            g_values = [a * (b + (1 - b) * (z**c)) for a, b, c in self._abc_tuples]
        g_min, g_max = min(g_values), max(g_values)
        f_min, f_max = self.fx_sign * self.fx_scale * g_min, self.fx_sign * self.fx_scale * g_max
        return f_min, f_max

    # -------------------------------------------------------------------------
    #  Final property computation
    # -------------------------------------------------------------------------
    @cached_property
    def ill_behaved(self) -> float:
        """
        0 if curve fitting produced a perfect fit (with 0 error), or 1.0 if |e| ~= |f(x)|
        The larger the relative error, the more pronounced are the properties of the function around the root
        that cannot be capture with our fitting curve.
        """

        # generate predictions based on optimal (a,b,c)
        a_opt, b_opt, c_opt = self._abc_opt
        gx_values = fitting_curve(self.x_pre, a_opt, b_opt, c_opt)
        fx_values_fit = self.fx_sign * self.fx_scale * gx_values

        # generate norms (based on avg(abs(.))) of error &
        norm_curve = np.average(np.abs(fx_values_fit))
        norm_error = np.average(np.abs(fx_values_fit - self.fx_values))

        # map to score
        if norm_curve == 0.0:
            # this is a degenerate case with all fx values 0 according to fit (i.e. a_opt=0)
            return 1.0
        else:
            # regular case
            return clip_scalar(norm_error / norm_curve, 0.0, 1.0)

    @cached_property
    def deriv_zero(self) -> float:
        # Only consider c>1 values if the uncertainty range has evidence for it.
        c = self._c_min_max[0]  # c = c_min
        # Map c --> score
        #         c <=1.0    -->        score = 0.0
        #   1.0 < c < 2.0    -->  0.0 < score < 1.0
        #         c >=2.0    -->        score = 1.0
        return math.log2(clip_scalar(c, 1.0, 2.0))

    @cached_property
    def deriv_infinite(self) -> float:
        # Only consider c<1 values if the uncertainty range has evidence for it.
        c = self._c_min_max[1]  # c = c_max
        # Map c --> score
        #         c <=0.5    -->        score = 1.0
        #   0.5 < c < 1.0    -->  1.0 > score > 0.0
        #         c >=1.0    -->        score = 0.0
        return -math.log2(clip_scalar(c, 0.5, 1.0))

    @cached_property
    def discontinuous(self) -> float:
        # Only consider b>0 values if the uncertainty range has evidence for it
        b = self._b_min_max[0]  # b = b_min
        # Map b --> score
        #         b <=0.0    -->        score = 1.0
        #   0.0 < b < 1.0    -->  0.0 < score < 1.0
        #         b >=1.0    -->        score = 0.0
        return clip_scalar(b, 0.0, 1.0)

    @cached_property
    def f0(self) -> tuple[float, float]:
        """Return (min,max)-values for estimated f(root) based on (a,b,c)-params in uncertainty range."""
        return self._f_min_max(z=0.0)

    @cached_property
    def f1(self) -> tuple[float, float]:
        """Return (min,max)-values for estimated f(root + 1*dx) based on (a,b,c)-params in uncertainty range."""
        return self._f_min_max(z=0.5)

    @cached_property
    def f2(self) -> tuple[float, float]:
        """Return (min,max)-values for estimated f(root + 2*dx) based on (a,b,c)-params in uncertainty range."""
        return self._f_min_max(z=1.0)

    @cached_property
    def f4(self) -> tuple[float, float]:
        """Return (min,max)-values for estimated f(root + 4*dx) based on (a,b,c)-params in uncertainty range."""
        return self._f_min_max(z=2.0)
