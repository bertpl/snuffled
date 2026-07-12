import pytest

from snuffled import Snuffler
from snuffled._core.analysis import FunctionSampler
from snuffled._core.analysis.roots.roots_analyser import RootsAnalyser
from snuffled._core.utils.constants import MIN_USEFUL_N_SAMPLES


def _f(x: float) -> float:
    return x - 0.3


# =================================================================================================
#  Sample counts below MIN_USEFUL_N_SAMPLES are rejected with a clear, parameter-named message
# =================================================================================================
@pytest.mark.parametrize("n", [0, 1, MIN_USEFUL_N_SAMPLES - 1])
def test_function_sampler_rejects_low_n_fun_samples(n: int):
    with pytest.raises(ValueError, match="n_fun_samples"):
        FunctionSampler(fun=_f, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=n, n_roots=100)


@pytest.mark.parametrize("n", [0, 1, MIN_USEFUL_N_SAMPLES - 1])
def test_function_sampler_rejects_low_n_roots(n: int):
    with pytest.raises(ValueError, match="n_roots"):
        FunctionSampler(fun=_f, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=100, n_roots=n)


@pytest.mark.parametrize("n", [0, 1, MIN_USEFUL_N_SAMPLES - 1])
def test_roots_analyser_rejects_low_n_root_samples(n: int):
    sampler = FunctionSampler(fun=_f, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=100, n_roots=100)
    with pytest.raises(ValueError, match="n_root_samples"):
        RootsAnalyser(sampler, n_root_samples=n, seed=42)


def test_min_useful_value_is_accepted():
    # exactly at the floor is valid (the message says ">= MIN_USEFUL_N_SAMPLES")
    _ = Snuffler(
        fun=_f,
        x_min=-1.0,
        x_max=1.0,
        dx=1e-9,
        seed=42,
        n_fun_samples=MIN_USEFUL_N_SAMPLES,
        n_roots=MIN_USEFUL_N_SAMPLES,
        n_root_samples=MIN_USEFUL_N_SAMPLES,
    )
