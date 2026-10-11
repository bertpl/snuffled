import numpy as np
import pytest

from snuffled._core.utils.rng import new_rng_state
from snuffled._core.utils.rng_seed import RngSeed


# ==================================================================================================
#  Pinned values: a change to the derivation changes every result of snuffled
# ==================================================================================================
@pytest.mark.parametrize(
    "labels, expected_value",
    [
        (("a",), 2311174872691823672),
        (("root", 0.25), 714859973596540386),
        (("a", 1), -5003063752518303446),
    ],
)
def test_rng_seed_derive_pinned(labels: tuple[str | int | float, ...], expected_value: int):
    """Deriving from seed 42 gives the pinned child seeds."""
    # --- act --------------------------
    child = RngSeed(42).derive(*labels)

    # --- assert -----------------------
    assert child.value == expected_value


# ==================================================================================================
#  derive
# ==================================================================================================
def test_rng_seed_derive_is_deterministic():
    """The same parent and labels give the same child."""
    # --- act / assert -----------------
    assert RngSeed(7).derive("x", 3, 0.5) == RngSeed(7).derive("x", 3, 0.5)


@pytest.mark.parametrize(
    "labels_1, labels_2",
    [
        (("a",), ("b",)),
        (("a", "b"), ("b", "a")),
        (("ab",), ("a", "b")),
        (("1",), (1,)),
        ((1,), (1.0,)),
        ((0.0,), (-0.0,)),
        ((), ("",)),
    ],
)
def test_rng_seed_derive_separates_labels(
    labels_1: tuple[str | int | float, ...], labels_2: tuple[str | int | float, ...]
):
    """Label lists that differ in value, order, split or type give different children."""
    # --- act / assert -----------------
    assert RngSeed(7).derive(*labels_1) != RngSeed(7).derive(*labels_2)


def test_rng_seed_derive_depends_on_parent():
    """The same labels under different parents give different children."""
    # --- act / assert -----------------
    assert RngSeed(1).derive("a") != RngSeed(2).derive("a")


@pytest.mark.parametrize(
    "label, error",
    [
        (True, TypeError),
        (None, TypeError),
        (b"a", TypeError),
        (np.float32(1.0), TypeError),
        (2**63, ValueError),
    ],
)
def test_rng_seed_derive_rejects_invalid_labels(label: object, error: type[Exception]):
    """A label that is not a str, int or float, or an int outside the int64 range, raises."""
    # --- act / assert -----------------
    with pytest.raises(error):
        RngSeed(7).derive(label)  # ty: ignore[invalid-argument-type]


# ==================================================================================================
#  Construction and generator state
# ==================================================================================================
@pytest.mark.parametrize("value", [-(2**63), 0, 2**63 - 1])
@pytest.mark.parametrize("labels", [(), ("x",), ("root", 0.25)])
def test_rng_seed_new_rng_state_matches_rng(value: int, labels: tuple[str | int | float, ...]):
    """The generator state for some labels is the one that `new_rng_state` builds from the child seed's value."""
    # --- act / assert -----------------
    expected_rng_state = new_rng_state(RngSeed(value).derive(*labels).value)
    assert np.array_equal(RngSeed(value).new_rng_state(*labels), expected_rng_state)


@pytest.mark.parametrize(
    "value, error",
    [
        (2**63, ValueError),
        (-(2**63) - 1, ValueError),
        (1.0, TypeError),
        (True, TypeError),
    ],
)
def test_rng_seed_rejects_invalid_values(value: object, error: type[Exception]):
    """A value outside the int64 range, or not an int, raises."""
    # --- act / assert -----------------
    with pytest.raises(error):
        RngSeed(value)  # ty: ignore[invalid-argument-type]
