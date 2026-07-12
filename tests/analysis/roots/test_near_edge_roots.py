import numpy as np
import pytest

from snuffled import Diagnostic, Snuffler
from snuffled._core.analysis import FunctionSampler
from snuffled._core.models import RootProperty


# =================================================================================================
#  Test functions
# =================================================================================================
def _mixed_roots(x: float) -> float:
    # roots at -1 + 2e-9 (near edge) plus 0.0 and 0.5 (interior)
    return (x - (-1.0 + 2e-9)) * x * (x - 0.5)


# =================================================================================================
#  analyzable_roots(): near-edge roots excluded, interior roots kept
# =================================================================================================
def test_analyzable_roots_excludes_near_edge_root():
    # --- arrange / act -----------------------------------
    # root at -1 + 2e-9, well inside the 4*sqrt(2)*dx (~= 5.66e-9) margin
    sampler = FunctionSampler(
        fun=lambda x: x - (-1.0 + 2e-9), x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=1000, n_roots=100
    )

    # --- assert ------------------------------------------
    assert len(sampler.roots()) > 0  # the root IS detected ...
    assert len(sampler.analyzable_roots()) == 0  # ... but excluded from two-sided analysis


def test_analyzable_roots_keeps_interior_root():
    # --- arrange / act -----------------------------------
    sampler = FunctionSampler(
        fun=lambda x: x - 0.3, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=1000, n_roots=100
    )

    # --- assert ------------------------------------------
    assert len(sampler.analyzable_roots()) == len(sampler.roots()) == 1


# =================================================================================================
#  Pipeline: near-edge roots no longer crash + diagnostic / fallback (regression R2)
# =================================================================================================
@pytest.mark.parametrize("root_x", [-1.0 + 2e-9, 1.0 - 2e-9])  # both edges
def test_extract_all_near_edge_root_completes(root_x: float):
    # --- act ---------------------------------------------
    props = Snuffler(
        fun=lambda x: x - root_x,
        x_min=-1.0,
        x_max=1.0,
        dx=1e-9,
        seed=42,
        n_fun_samples=1000,
        n_roots=100,
        n_root_samples=100,
    ).extract_all()

    # --- assert ------------------------------------------
    assert np.all(np.isfinite(props.as_array()))
    assert props[Diagnostic.ALL_ROOTS_TOO_CLOSE_TO_EDGE] == 1.0  # only root is near an edge
    assert all(props[rp] == 0.0 for rp in RootProperty)  # no analyzable roots -> 0.0 fallback


def test_extract_all_mixed_roots_analyzes_interior():
    # --- act ---------------------------------------------
    props = Snuffler(
        fun=_mixed_roots, x_min=-1.0, x_max=1.0, dx=1e-9, seed=42, n_fun_samples=1000, n_roots=100, n_root_samples=100
    ).extract_all()

    # --- assert ------------------------------------------
    assert np.all(np.isfinite(props.as_array()))
    assert props[Diagnostic.ALL_ROOTS_TOO_CLOSE_TO_EDGE] == 0.0  # interior roots ARE analyzable
