import sys

# --- accuracy --------------------------------------------
EPS = sys.float_info.epsilon  # 2**-52 for float64

# --- seed offsets ----------------------------------------
# these offsets are used in different functions to ensure
# we do not use the same seed in each, even when set
# manually in the top-most function.

# utils.sampling
SEED_OFFSET_MULTI_SCALE_SAMPLES = 2_427_111
SEED_OFFSET_SAMPLE_INTEGERS = 4_911_514
SEED_OFFSET_PSEUDO_UNIFORM_SAMPLES = 9_909_219
