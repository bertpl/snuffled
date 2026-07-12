import math
from collections.abc import Callable

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from snuffled import Diagnostic, FunctionProperty, RootProperty, Snuffler
from tests.helpers import only_with_numba_jit


# =================================================================================================
#  Function zoo: numerically hostile but type-valid callables, all defined on [-1, 1]
# =================================================================================================
def _zoo() -> list[Callable[[float], float]]:
    return [
        lambda x: 1.0 + x * x,  # rootless parabola
        lambda x: 1.0,  # constant
        lambda x: 0.0,  # all zeros
        lambda x: x,  # simple root at 0
        lambda x: x - 0.9999,  # near-edge root
        lambda x: float(np.sign(x)),  # step (bounded discontinuity)
        lambda x: math.sin(1000.0 * x),  # many roots
        lambda x: 10.0 ** (20.0 * x),  # high dynamic range
        lambda x: float("nan") if abs(x) < 0.3 else x,  # NaN over a region
        lambda x: float("inf") if x > 0.5 else x,  # +inf over a region
        lambda x: 1.0 / (x - 0.2) if x != 0.2 else float("inf"),  # pole
        lambda x: 1e-300 * (x - 0.1),  # underflow-prone
    ]


_ZOO = _zoo()
_DX = st.sampled_from([1e-3, 1e-6, 1e-9, 1e-12])
_SEED = st.integers(min_value=0, max_value=2**31 - 1)
_N = st.integers(min_value=10, max_value=200)  # kept small so whole-pipeline runs stay fast in CI


@st.composite
def _params(draw: st.DrawFn) -> dict:
    return {
        "fun": draw(st.sampled_from(_ZOO)),
        "x_min": -1.0,
        "x_max": 1.0,
        "dx": draw(_DX),
        "seed": draw(_SEED),
        "n_fun_samples": draw(_N),
        "n_roots": draw(_N),
        "n_root_samples": draw(_N),
    }


# =================================================================================================
#  Properties 1 + 2: no crash, and every property stays in its documented range
# =================================================================================================
@only_with_numba_jit
@given(params=_params())
def test_no_crash_and_output_in_domain(params: dict) -> None:
    # property 1 (no crash): simply reaching the assertions below.
    props = Snuffler(**params).extract_all()

    # property 2 (output domain): scores in [0, 1]; MAX_ZERO_WIDTH is a raw width in [0, interval].
    for fp in FunctionProperty:
        assert 0.0 <= props[fp] <= 1.0
    for rp in RootProperty:
        assert 0.0 <= props[rp] <= 1.0
    interval = params["x_max"] - params["x_min"]
    for d in Diagnostic:
        if d == Diagnostic.MAX_ZERO_WIDTH:
            assert 0.0 <= props[d] <= interval
        else:
            assert 0.0 <= props[d] <= 1.0


# =================================================================================================
#  Property 3: same (fun, params, seed) -> identical output (all zoo functions are deterministic)
# =================================================================================================
@only_with_numba_jit
@given(params=_params())
def test_determinism(params: dict) -> None:
    first = Snuffler(**params).extract_all().as_array()
    second = Snuffler(**params).extract_all().as_array()
    assert np.array_equal(first, second)
