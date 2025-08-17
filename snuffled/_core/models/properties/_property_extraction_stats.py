from dataclasses import dataclass


@dataclass(frozen=True)
class PropertyExtractionStats:
    property: str
    t_start_ns: int  # start of property extraction based on time_ns()
    t_duration_sec: float  # duration of property extraction based delta between 2 perf_counter_ns() calls
    n_f_samples: int  # number of function samples used (on top of what was cached from before)
