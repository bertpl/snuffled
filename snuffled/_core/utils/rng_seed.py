"""This module derives the seeds of independent random draws from 1 user seed, by hashing labels into it.

Each random draw in snuffled builds its own generator state from a seed that is derived from the
user's seed through a chain of labels, such as `("function_sampler",)` then `("grid",)`. A draw's
random numbers therefore depend only on the user's seed and its own chain of labels: not on which
other draws happened before it, nor on how many.
"""

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from .rng import new_rng_state

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class RngSeed:
    """A 64-bit seed, from which child seeds and generator states are derived.

    A child seed is a hash of its parent's value and a list of labels, so 2 children whose label lists
    differ get unrelated seeds. The hash is blake2b, not Python's built-in `hash()`, which gives
    different values for the same string in different processes.
    """

    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise TypeError(f"seed must be an int, got {type(self.value).__name__}")
        if not (_INT64_MIN <= self.value <= _INT64_MAX):
            raise ValueError(f"seed must fit in an int64, got {self.value}")

    # --------------------------------------------------------------------------
    #  Main API
    # --------------------------------------------------------------------------
    def derive(self, *labels: str | int | float) -> "RngSeed":
        """Return the child seed of this seed for the given labels.

        Each label is a `str`, an `int` that fits in an int64, or a `float`, which is hashed by its
        64-bit pattern. The same labels in the same order always give the same child.
        """
        digest = hashlib.blake2b(digest_size=8)
        digest.update(struct.pack("<q", self.value))
        for label in labels:
            digest.update(self._encode_label(label))
        return RngSeed(int.from_bytes(digest.digest(), "little", signed=True))

    def new_rng_state(self) -> np.ndarray:
        """Return a new generator state for this seed, as `rng.new_rng_state` builds it."""
        return new_rng_state(self.value)

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
            return b"i" + struct.pack("<q", label)
        elif isinstance(label, float):
            return b"f" + struct.pack("<d", label)
        else:
            raise TypeError(f"label must be a str, int or float, got {type(label).__name__}")
