import math
import sys

# --- accuracy --------------------------------------------
EPS = sys.float_info.epsilon  # 2**-52 for float64

# f(x) magnitudes are clipped to +/- FX_CLIP at the sampling boundary so +/-inf never reaches the
# analysis kernels (inf - inf = nan would otherwise poison them). Invariant: FX_CLIP**2 == eps*max,
# i.e. a squared clipped value sits a full 52-bit mantissa below overflow -- safe under squaring and
# summation. Far above where high_dynamic_range saturates (q90/q10 = 2**94), so clipping is invisible
# to that metric for any sane function.
FX_CLIP = math.sqrt(sys.float_info.max) * math.sqrt(EPS)

# --- sample counts ---------------------------------------
# Public default sample counts (used by Snuffler's signature; tests import these to build
# at-default samplers without hard-coding the values).
DEFAULT_N_FUN_SAMPLES = 10_000
DEFAULT_N_ROOTS = 100
DEFAULT_N_ROOT_SAMPLES = 100

# Hard floor for every sample-count parameter: below this the analysis is not statistically useful
# (and the sampling utilities have no headroom); constructors reject smaller values.
MIN_USEFUL_N_SAMPLES = 10

# --- seed offsets ----------------------------------------
# these offsets are used in different functions to ensure
# we do not use the same seed in each, even when set
# manually in the top-most function.

# property extractors & co
SEED_OFFSET_SNUFFLER = 3_061_002
SEED_OFFSET_FUNCTION_SAMPLER = 3_541_253
SEED_OFFSET_ROOTS_ANALYSER = 2_389_859
SEED_OFFSET_SINGLE_ROOT_ANALYSER = 6_005_595

# curve_fitting
SEED_OFFSET_COMPUTE_X_DELTAS = 1_321_350

# utils.sampling
SEED_OFFSET_MULTI_SCALE_SAMPLES = 2_427_111
SEED_OFFSET_SAMPLE_INTEGERS = 4_911_514
SEED_OFFSET_PSEUDO_UNIFORM_SAMPLES = 9_909_219
