from typing import TYPE_CHECKING

import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models import SnuffledRootProperties
from snuffled._core.utils.rng_seed import RngSeed
from snuffled._core.utils.validation import require_min_n_samples

from .single_root_two_side_analyser import SingleRootTwoSideAnalyser

if TYPE_CHECKING:
    from snuffled._core.models.root_analysis import Root


class RootsAnalyser(PropertyExtractor[SnuffledRootProperties]):
    # --------------------------------------------------------------------------
    #  Constructor
    # --------------------------------------------------------------------------
    def __init__(self, function_sampler: FunctionSampler, n_root_samples: int, rng_seed: RngSeed) -> None:
        super().__init__(function_sampler)
        require_min_n_samples("n_root_samples", n_root_samples)
        self.n_root_samples = n_root_samples
        self._root_analyses: dict[Root, SnuffledRootProperties] = {}
        self._rng_seed = rng_seed.get_derived_seed("roots_analyser")

    # --------------------------------------------------------------------------
    #  Main Implementation
    # --------------------------------------------------------------------------
    def _new_named_array(self) -> SnuffledRootProperties:
        return SnuffledRootProperties()

    def _extract(self, prop: str) -> float:
        self._ensure_all_roots_analysed()

        # compute overall score
        if prop in self.supported_properties():
            # take average of this property over all analysed roots
            if len(self._root_analyses) > 0:
                return float(np.mean([root_props[prop] for root_props in self._root_analyses.values()]))
            else:
                # Without roots every root property is 0.0; Diagnostic.NO_ZEROS_DETECTED flags this case.
                return 0.0
        else:
            raise ValueError(f"Property {prop} not supported")

    # --------------------------------------------------------------------------
    #  Internal methods
    # --------------------------------------------------------------------------
    def _ensure_all_roots_analysed(self) -> None:
        # only roots with room for the full sampling span on both sides — near-edge roots can't be
        # reliably two-side-fitted and are flagged by the ALL_ROOTS_TOO_CLOSE_TO_EDGE diagnostic.
        roots = self.function_sampler.analyzable_roots
        if len(self._root_analyses) < len(roots):
            self._root_analyses = {
                root: SingleRootTwoSideAnalyser(
                    self.function_sampler,
                    root,
                    self.n_root_samples,
                    # a root's seed depends on the root's own x, so the seed does not change when other roots
                    # are added to or dropped from the analysis
                    self._rng_seed.get_derived_seed("root", root.x),
                ).extract_all()
                for root in roots
            }
