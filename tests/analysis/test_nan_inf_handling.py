import math
from collections.abc import Callable

import numpy as np
import pytest

from snuffled import Diagnostic, FunctionProperty, Snuffler
from snuffled._core.analysis import FunctionSampler
from snuffled._core.utils.constants import FX_CLIP


# ==================================================================================================
#  Test functions
# ==================================================================================================
def _f_nan_middle(x: float) -> float:
    # NaN over a region
    if abs(x) < 0.3:
        return float("nan")
    else:
        return x


def _f_nan_at_only_root(x: float) -> float:
    # NaN covers the only sign flip
    if abs(x) < 0.3:
        return float("nan")
    elif x > 0:
        return x - 0.5
    else:
        return x


def _f_inf_region(x: float) -> float:
    # +inf over a region -> the grid samples it
    if x > 0.5:
        return float("inf")
    else:
        return x


def _f_pole(x: float) -> float:
    if x != 0.2:
        return 1.0 / (x - 0.2)
    else:
        return float("inf")


# ==================================================================================================
#  Boundary sanitize (FunctionSampler.f): NaN / +-inf never leave f(); flags are set
# ==================================================================================================
def test_f_clips_positive_inf_and_flags():
    # --- arrange / act / assert -------
    sampler = FunctionSampler(
        n_roots=100, fun=lambda x: float("inf"), x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=100
    )
    assert sampler.f(0.0) == FX_CLIP
    assert sampler.saw_inf
    assert not sampler.saw_nan


def test_f_clips_negative_inf():
    sampler = FunctionSampler(
        n_roots=100, fun=lambda x: float("-inf"), x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=100
    )
    assert sampler.f(0.0) == -FX_CLIP
    assert sampler.saw_inf


def test_f_sanitizes_nan_and_flags():
    sampler = FunctionSampler(
        n_roots=100, fun=lambda x: float("nan"), x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=100
    )
    assert math.isfinite(sampler.f(0.0))
    assert sampler.saw_nan
    assert not sampler.saw_inf


def test_f_clips_huge_finite_without_inf_flag():
    sampler = FunctionSampler(
        n_roots=100, fun=lambda x: FX_CLIP * 10, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=100
    )
    assert sampler.f(0.0) == FX_CLIP  # clipped for arithmetic safety ...
    assert not sampler.saw_inf  # ... but a finite value is not flagged as inf
    assert not sampler.saw_nan


# ==================================================================================================
#  Pipeline: extract_all() completes on functions that return NaN or inf, and flags those values
# ==================================================================================================
@pytest.mark.parametrize("fun", [_f_nan_middle, _f_nan_at_only_root])
def test_extract_all_nan_completes_and_flags(fun: Callable[[float], float]):
    # --- act --------------------------
    props = Snuffler(fun=fun, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42).extract_all()

    # --- assert -----------------------
    assert np.all(np.isfinite(props.as_array()))  # never crashes, output stays finite
    assert props[Diagnostic.NAN_VALUES_DETECTED] == 1.0


def test_extract_all_inf_completes_and_flags():
    # --- act --------------------------
    props = Snuffler(fun=_f_inf_region, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42).extract_all()

    # --- assert -----------------------
    assert np.all(np.isfinite(props.as_array()))
    assert props[Diagnostic.INF_VALUES_DETECTED] == 1.0


def test_extract_all_pole_has_finite_discontinuity():
    # without clipping +-inf to +-FX_CLIP, the pole would make the discontinuity score NaN
    # --- act --------------------------
    props = Snuffler(fun=_f_pole, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42).extract_all()

    # --- assert -----------------------
    assert np.all(np.isfinite(props.as_array()))
    assert props[FunctionProperty.DISCONTINUOUS] > 0.99


# ==================================================================================================
#  Deep detection: a non-finite value first hit during root-finding (not the initial grid) is flagged
# ==================================================================================================
@pytest.mark.parametrize(
    "bad, diagnostic",
    [
        (float("nan"), Diagnostic.NAN_VALUES_DETECTED),
        (float("inf"), Diagnostic.INF_VALUES_DETECTED),
    ],
)
def test_non_finite_hit_only_in_deep_analysis_is_flagged(bad: float, diagnostic: Diagnostic):
    # A tiny non-finite band around the root at 0: a coarse grid (n_fun_samples=10) misses it, but
    # root-finding bisects into it -> the diagnostic must reflect that deeper sampling, not the grid alone.
    def fun(x: float) -> float:
        if abs(x) < 1e-7:
            return bad
        else:
            return x

    # --- act --------------------------
    props = Snuffler(
        fun=fun, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=10, n_roots=10, n_root_samples=10
    ).extract_all()

    # --- assert -----------------------
    assert props[diagnostic] == 1.0
