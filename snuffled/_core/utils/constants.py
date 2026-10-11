import math
import sys

# ==================================================================================================
#  Accuracy
# ==================================================================================================
EPS = sys.float_info.epsilon  # 2**-52 for float64

# f(x) magnitudes are clipped to +/- FX_CLIP at the sampling boundary so +/-inf never reaches the
# analysis kernels (inf - inf = nan would otherwise poison them). Invariant: FX_CLIP**2 == eps*max,
# i.e. a squared clipped value sits a full 52-bit mantissa below overflow -- safe under squaring and
# summation. Far above where high_dynamic_range saturates (q90/q10 = 2**94), so clipping is invisible
# to that metric for any sane function.
FX_CLIP = math.sqrt(sys.float_info.max) * math.sqrt(EPS)

# ==================================================================================================
#  Sample counts
# ==================================================================================================
# Public default sample counts (used by Snuffler's signature; tests import these to build
# at-default samplers without hard-coding the values).
DEFAULT_N_FUN_SAMPLES = 10_000
DEFAULT_N_ROOTS = 100
DEFAULT_N_ROOT_SAMPLES = 100

# Hard floor for every sample-count parameter: below this the analysis is not statistically useful
# (and the sampling utilities have no headroom); constructors reject smaller values.
MIN_USEFUL_N_SAMPLES = 10
