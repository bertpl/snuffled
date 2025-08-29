import numpy as np

from snuffled._core.analysis._function_sampler import FunctionSampler
from snuffled._core.analysis._property_extractor import PropertyExtractor
from snuffled._core.models import SnuffledRootProperties
from snuffled._core.utils.constants import SEED_OFFSET_ROOTS_ANALYSER

from .single_root import SingleRootAnalyser


class RootsAnalyser(PropertyExtractor[SnuffledRootProperties]):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, function_sampler: FunctionSampler, dx: float, n_root_samples: int, seed: int = 42):
        super().__init__(function_sampler)
        self.dx = dx
        self.n_root_samples = n_root_samples
        self._root_analyses: dict[tuple[float, float], SnuffledRootProperties] = dict()
        self._seed = seed + SEED_OFFSET_ROOTS_ANALYSER

    # -------------------------------------------------------------------------
    #  Main Implementation
    # -------------------------------------------------------------------------
    def _new_named_array(self) -> SnuffledRootProperties:
        return SnuffledRootProperties()

    def _extract(self, prop: str) -> float:
        # make sure we have analysed all roots provided by function_sampler.roots()
        # such that self._root_analyses is populated
        self._ensure_all_roots_analysed()

        # compute overall score
        if prop in self.supported_properties():
            # take average of this property over all analysed roots
            return float(np.mean([root_props[prop] for root_props in self._root_analyses.values()]))
        else:
            raise ValueError(f"Property {prop} not supported")

    # -------------------------------------------------------------------------
    #  Internal methods
    # -------------------------------------------------------------------------
    def _ensure_all_roots_analysed(self):
        if not self._root_analyses:
            self._root_analyses = {
                root: SingleRootAnalyser(
                    self.function_sampler,
                    root,
                    self.dx,
                    self.n_root_samples,
                    self._seed + (i * SEED_OFFSET_ROOTS_ANALYSER),
                ).extract_all()
                for i, root in enumerate(self.function_sampler.roots())
            }
