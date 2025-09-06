from abc import ABC, abstractmethod
from time import perf_counter_ns, time_ns
from typing import Generic, TypeVar

from snuffled._core.models import NamedArray, PropertyExtractionStats

from ._function_sampler import FunctionSampler

NA = TypeVar("NA", bound=NamedArray)


class PropertyExtractor(ABC, Generic[NA]):
    # =================================================================================================
    #  Main API
    # =================================================================================================
    def __init__(self, function_sampler: FunctionSampler):
        self.function_sampler = function_sampler
        self.__stats: dict[str, PropertyExtractionStats] = dict()  # property -> stats

    def supported_properties(self) -> list[str]:
        """Return list of all supported properties, in order in which they should be extracted by extract_all"""
        return self._new_named_array().names()

    def extract_all(self) -> NA:
        """Extract all properties and returned as specific NamedArray subclass."""
        named_array = self._new_named_array()
        for property_name in self.supported_properties():
            named_array[property_name] = self.extract(property_name)
        return named_array

    def extract(self, prop: str) -> float:
        """Extract specific property, while tracking time and duration of extraction."""

        # --- timed property extraction -------------------
        t_start_ns_epoch = time_ns()  # ns since epoch
        t_start_ns = perf_counter_ns()
        n_f_samples_before = self.function_sampler.function_cache_size()
        result = self._extract(prop)
        t_end_ns = perf_counter_ns()
        n_f_samples_after = self.function_sampler.function_cache_size()

        # --- store stats ---------------------------------
        self.__stats[prop] = PropertyExtractionStats(
            property=prop,
            t_start_ns=t_start_ns_epoch,
            t_duration_sec=(t_end_ns - t_start_ns) / 1e9,
            n_f_samples=n_f_samples_after - n_f_samples_before,
        )

        # --- return result -------------------------------
        return result

    def statistics(self) -> dict[str, PropertyExtractionStats]:
        """
        Return dictionary with property extraction stats by property name.  Entries are sorted by
        order of extraction.
        """
        return dict(
            sorted(
                self.__stats.items(),
                key=lambda prop_stat: prop_stat[1].t_start_ns,
            )
        )

    # =================================================================================================
    #  Abstract methods
    # =================================================================================================
    @abstractmethod
    def _extract(self, prop: str) -> float:
        """Extract specific property."""
        raise NotImplementedError()

    @abstractmethod
    def _new_named_array(self) -> NA:
        raise NotImplementedError()
