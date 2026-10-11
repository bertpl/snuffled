import numpy as np
import pytest

from snuffled._core.utils.statistics._estimate_sign_flip_frequency import cross_entropy_loss


@pytest.mark.parametrize(
    "w, p, lamb",
    [
        ([0.0, 0.5], [0.0, 1.0], 0.55),  # a zero-width interval without a sign flip
        ([1e-17, 0.5], [0.0, 1.0], 0.55),  # exp(-2*lamb*w) rounds to 1.0, without a sign flip
        ([1e-17, 0.5], [1.0, 0.0], 0.55),  # exp(-2*lamb*w) rounds to 1.0, with a sign flip
        ([1e-300, 0.5], [0.5, 0.5], 1e10),  # a probability between 0 and 1
    ],
)
def test_cross_entropy_loss_is_finite(w: list[float], p: list[float], lamb: float):
    """cross_entropy_loss stays finite for zero and vanishingly small interval widths."""
    # --- act --------------------------
    loss = cross_entropy_loss(np.array(w), np.array(p), lamb)

    # --- assert -----------------------
    assert np.isfinite(loss)
