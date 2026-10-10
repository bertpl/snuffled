class NamedArray:
    """Array of numbers that can also be accessed by means of str-valued identifiers, much like a dict."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, names: list[str], values: list[float] | None = None) -> None:
        """Initialize the NamedArray with names and values.

        Args:
            names: Names of the array elements; each name may appear only once. The array keeps this list, not a copy.
            values: Values of the array elements, one per name. `None` gives all zeros. The array keeps this list,
                not a copy, so setting an element also changes the caller's list.

        Raises:
            ValueError: If a name appears more than once, or if `values` does not hold exactly one value per name.
        """
        if len(set(names)) != len(names):
            raise ValueError(f"Names must be unique; got {names}")
        if values is None:
            values = [0.0] * len(names)
        elif len(values) != len(names):
            raise ValueError(f"Expected {len(names)} values, one per name; got {len(values)}")
        self._names = names
        self._values = values

    # -------------------------------------------------------------------------
    #  Conversion methods
    # -------------------------------------------------------------------------
    def names(self) -> list[str]:
        return self._names.copy()

    def as_array(self) -> list[float]:
        return self._values.copy()

    def as_dict(self) -> dict[str, float]:
        """Return a dict that maps each name to its value."""
        return dict(zip(self._names, self._values))

    # -------------------------------------------------------------------------
    #  Overridden methods
    # -------------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        return isinstance(other, NamedArray) and (self._names == other._names) and (self._values == other._values)

    # mutable container (see __setitem__): intentionally unhashable
    __hash__ = None

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: str | int) -> float:
        return self._values[self._key_to_index(key)]

    def __setitem__(self, key: str | int, value: float) -> None:
        self._values[self._key_to_index(key)] = value

    # -------------------------------------------------------------------------
    #  Internals
    # -------------------------------------------------------------------------
    def _key_to_index(self, key: str | int) -> int:
        """Return the index for a name or an int index; an int is returned as is, without a range check.

        Raises:
            KeyError: If `key` is a name that is not in the array.
            TypeError: If `key` is neither a str nor an int.
        """
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            if key in self._names:
                return self._names.index(key)
            else:
                raise KeyError(f"Name '{key}' not found in NamedArray.")
        else:
            raise TypeError(f"Invalid key type {type(key)}")
