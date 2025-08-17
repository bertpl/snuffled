class NamedArray:
    """
    Class representing an array of numbers that can also be accessed by means of str-valued identifiers, pretty
    much like a dict.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, names: list[str], values: list[float] | None = None):
        """
        Initialize the NamedArray with names and values.

        :param names: List of names (str) for the array elements.
        :param values: List of values (float) corresponding to the names.
        """
        self._names = names
        self._values = values or [0.0] * len(names)

    # -------------------------------------------------------------------------
    #  Conversion methods
    # -------------------------------------------------------------------------
    def names(self) -> list[str]:
        return self._names.copy()

    def as_array(self) -> list[float]:
        return self._values.copy()

    def as_dict(self) -> dict[str, float]:
        """
        Convert the NamedArray to a dictionary with names as keys and values as values.
        :return: Dictionary representation of the NamedArray.
        """
        return {name: value for name, value in zip(self._names, self._values)}

    # -------------------------------------------------------------------------
    #  Overridden methods
    # -------------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        return isinstance(other, NamedArray) and (self._names == other._names) and (self._values == other._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key):
        return self._values[self._key_to_index(key)]

    def __setitem__(self, key, value):
        self._values[self._key_to_index(key)] = value

    # -------------------------------------------------------------------------
    #  Internals
    # -------------------------------------------------------------------------
    def _key_to_index(self, key) -> int:
        """
        Convert a key (name or index) to an index.
        :param key: The key to convert.
        :return: The index corresponding to the key.
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
