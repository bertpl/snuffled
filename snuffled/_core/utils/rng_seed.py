"""Every random draw gets its own seed, derived from 1 user seed by hashing labels into the user seed.

Each random draw in snuffled builds its own generator state from a seed that is derived from the
user's seed by successive `derive` calls, each with its own labels, such as `derive("function_sampler")`
then `derive("grid")`.

A draw's random numbers therefore depend only on the user's seed and the labels of those `derive`
calls: not on which other draws happened before it, nor on how many.
"""

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from .rng import new_rng_state

_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class RngSeed:
    """An `RngSeed` holds a 64-bit seed, from which child seeds and generator states are derived.

    A child seed is a hash of its parent's value and a list of labels, so 2 children whose label lists
    differ get unrelated seeds. The hash is blake2b, not Python's built-in `hash()`, which gives
    different values for the same string in different processes.
    """

    value: int

    def __post_init__(self) -> None:
        """Reject a value that is not an `int`, or that does not fit in an int64."""
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise TypeError(f"seed must be an int, got {type(self.value).__name__}")
        if not (_INT64_MIN <= self.value <= _INT64_MAX):
            raise ValueError(f"seed must fit in an int64, got {self.value}")

    # --------------------------------------------------------------------------
    #  Main API
    # --------------------------------------------------------------------------
    def derive(self, *labels: str | int | float) -> "RngSeed":
        """Return the child seed of this seed for the given labels.

        The same labels in the same order always give the same child. Each label is one of:

        - a `str`;
        - an `int` that fits in an int64;
        - a `float`, which is hashed by its 64-bit pattern.

        Raises:
            TypeError: If a label is not a `str`, `int` or `float`; a `bool` is rejected too.
            ValueError: If an `int` label does not fit in an int64.
        """
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(struct.pack("<q", self.value))
        for label in labels:
            hasher.update(self._encode_label(label))
        return RngSeed(int.from_bytes(hasher.digest(), "little", signed=True))

    def derive_rng_state(self, *labels: str | int | float) -> np.ndarray:
        """Return a new generator state for the child seed of this seed for the given labels.

        Each random draw passes its own labels, so that no 2 draws share a generator state.
        """
        return new_rng_state(self.derive(*labels).value)

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _encode_label(label: str | int | float) -> bytes:
        """Encode a label as bytes that start with a type tag, so that labels of different types never collide.

        A `str` also carries its length, so that the encoding of a label list has only 1 way to split
        into labels.
        """
        if isinstance(label, str):
            encoded = label.encode("utf-8")
            return b"s" + struct.pack("<q", len(encoded)) + encoded
        elif isinstance(label, int) and not isinstance(label, bool):
            if not (_INT64_MIN <= label <= _INT64_MAX):
                raise ValueError(f"int label must fit in an int64, got {label}")
            return b"i" + struct.pack("<q", label)
        elif isinstance(label, float):
            return b"f" + struct.pack("<d", label)
        else:
            raise TypeError(f"label must be a str, int or float, got {type(label).__name__}")
