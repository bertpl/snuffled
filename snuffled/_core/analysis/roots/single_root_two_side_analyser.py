import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models import RootProperty, SnuffledRootProperties
from snuffled._core.models.root_analysis import Root
from snuffled._core.utils.constants import SEED_OFFSET_SINGLE_ROOT_ANALYSER
from snuffled._core.utils.numba import clip_scalar
from snuffled._core.utils.signs import robust_sign_estimate

from .curve_fitting import compute_x_deltas
from .single_root_one_side_analyser import SingleRootOneSideAnalyser


class SingleRootTwoSideAnalyser(PropertyExtractor[SnuffledRootProperties]):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        function_sampler: FunctionSampler,
        root: Root,
        dx: float,
        n_root_samples: int,
        seed: int,
    ):
        super().__init__(function_sampler)
        self.root = root
        self.dx = dx
        self.n_root_samples = n_root_samples
        self._seed = seed + SEED_OFFSET_SINGLE_ROOT_ANALYSER

        # actual analyses
        self._analysis_left: SingleRootOneSideAnalyser
        self._analysis_right: SingleRootOneSideAnalyser
        self._perform_analyses()

    # -------------------------------------------------------------------------
    #  Internal - Root Analysis
    # -------------------------------------------------------------------------
    def _perform_analyses(self):
        """Performs left- and right-sided root analysis and populates self._analysis_left, self._analysis_right."""

        # --- sample function around root -----------------
        x_deltas = compute_x_deltas(dx=self.dx, k=self.n_root_samples, seed=self._seed)
        fx_values_left = np.array([self.function_sampler.f(self.root.x - x_delta) for x_delta in x_deltas])
        fx_values_right = np.array([self.function_sampler.f(self.root.x + x_delta) for x_delta in x_deltas])

        # --- determine left/right function sign ----------

        # get robust sign estimates from left & right fx-values
        sign_estimate_left = robust_sign_estimate(fx_values_left)
        sign_estimate_right = robust_sign_estimate(fx_values_left)
        if sign_estimate_left * sign_estimate_right == -1:
            # In most cases this test should be definitive
            fx_sign_left = sign_estimate_left
            fx_sign_right = sign_estimate_right
        else:
            # fall back to what we determined while finding this root.  Since we only consider odd multiplicity roots
            # and ignore even roots, this will always be discriminative
            fx_sign_left = -self.root.deriv_sign
            fx_sign_right = self.root.deriv_sign

        # --- actual analyses -----------------------------
        self._analysis_left = SingleRootOneSideAnalyser(self.dx, x_deltas, fx_values_left, -1, fx_sign_left)
        self._analysis_right = SingleRootOneSideAnalyser(self.dx, x_deltas, fx_values_right, +1, fx_sign_right)

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def _new_named_array(self) -> SnuffledRootProperties:
        return SnuffledRootProperties()

    def _extract(self, prop: str) -> float:
        match prop:
            case RootProperty.ILL_BEHAVED:
                return self._extract_ill_behaved()
            case RootProperty.DERIVATIVE_ZERO:
                return self._extract_derivative_zero()
            case RootProperty.DERIVATIVE_INFINITE:
                return self._extract_derivative_infinite()
            case RootProperty.DISCONTINUOUS:
                return self._extract_discontinuous()
            case RootProperty.ASYMMETRIC:
                return self._extract_asymmetric()
            case _:
                raise ValueError(f"Property {prop} not supported")

    # -------------------------------------------------------------------------
    #  Internal methods
    # -------------------------------------------------------------------------
    def _extract_ill_behaved(self) -> float:
        return 0.5 * (self._analysis_left.ill_behaved + self._analysis_right.ill_behaved)

    def _extract_derivative_zero(self) -> float:
        return 0.5 * (self._analysis_left.deriv_zero + self._analysis_right.deriv_zero)

    def _extract_derivative_infinite(self) -> float:
        return 0.5 * (self._analysis_left.deriv_infinite + self._analysis_right.deriv_infinite)

    def _extract_discontinuous(self) -> float:
        return 0.5 * (self._analysis_left.discontinuous + self._analysis_right.discontinuous)

    def _extract_asymmetric(self) -> float:
        # --- aggregate norms & deltas of f0,f1,f2,f4 -----
        total_norm = 0.0
        total_delta = 0.0
        for att in ["f0", "f1", "f2", "f4"]:
            f_left_min, f_left_max = getattr(self._analysis_left, att)
            f_right_min, f_right_max = getattr(self._analysis_right, att)

            total_norm += max(abs(f_left_min), abs(f_left_max), abs(f_right_min), abs(f_right_max))
            total_delta += range_diff(f_left_min, f_left_max, -f_right_max, -f_right_min)

        # --- final score ----------------------------------
        if total_norm == 0.0:
            # this means both sides are 0
            return 0.0
        else:
            # regular case
            return clip_scalar(total_delta / total_norm, 0.0, 1.0)


# =================================================================================================
#  Helpers
# =================================================================================================
def range_diff(x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """
    Computes distance between range (x_min, x_max) and (y_min, y_max), as the absolute difference between
    closest two points.
    """
    if x_max < y_min:
        # (x_min, x_max) < (y_min, y_max)  --> closest 2 points are x_max, y_min
        return y_min - x_max
    elif y_max < x_min:
        # (y_min, y_max) < (x_min, x_max)  --> closest 2 points are y_max, x_min
        return x_min - y_max
    else:
        # intervals must overlap at least partially
        return 0.0
