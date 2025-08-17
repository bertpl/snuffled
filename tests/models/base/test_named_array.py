import pytest

from snuffled._core.compatibility import StrEnum
from snuffled._core.models.base import NamedArray


def test_named_array_construction_empty():
    # --- arrange -----------------------------------------
    names = ["a", "b", "c"]

    # --- act ---------------------------------------------
    named_array = NamedArray(names)

    # --- assert ------------------------------------------
    assert len(named_array) == 3
    assert named_array.names() == names
    assert named_array.as_array() == [0.0, 0.0, 0.0]
    assert named_array.as_dict() == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_named_array_construction_with_values():
    # --- arrange -----------------------------------------
    names = ["a", "b", "c"]
    values = [1.0, 2.0, 3.0]

    # --- act ---------------------------------------------
    named_array = NamedArray(names, values)

    # --- assert ------------------------------------------
    assert len(named_array) == 3
    assert named_array.names() == names
    assert named_array.as_array() == [1.0, 2.0, 3.0]
    assert named_array.as_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_named_array_construction_with_str_enum():
    # --- arrange -----------------------------------------
    class MyStrEnum(StrEnum):
        MEMBER_A = "a"
        MEMBER_B = "b"
        MEMBER_C = "c"

    values = [3.0, 5.0, 7.0]

    # --- act ---------------------------------------------
    named_array = NamedArray(list(MyStrEnum), values)

    # --- assert ------------------------------------------
    assert len(named_array) == 3
    assert named_array.names() == ["a", "b", "c"]
    assert named_array.names() == list(MyStrEnum)
    assert named_array.names() == [MyStrEnum.MEMBER_A, MyStrEnum.MEMBER_B, MyStrEnum.MEMBER_C]
    assert named_array.as_array() == [3.0, 5.0, 7.0]
    assert named_array.as_dict() == {"a": 3.0, "b": 5.0, "c": 7.0}


def test_named_array_get_set():
    # --- arrange -----------------------------------------
    named_array = NamedArray(["a", "b", "c"])

    # --- act ---------------------------------------------
    named_array["a"] = 1.0
    named_array["b"] = 2.0
    named_array["c"] = 3.0

    named_array[2] = 4.0

    # --- assert ------------------------------------------
    assert named_array["a"] == 1.0
    assert named_array["b"] == 2.0
    assert named_array["c"] == 4.0

    assert named_array[0] == 1.0
    assert named_array[1] == 2.0
    assert named_array[2] == 4.0


def test_named_array_get_error():
    # --- arrange -----------------------------------------
    named_array = NamedArray(["a", "b", "c"])

    # --- act & assert ------------------------------------
    with pytest.raises(IndexError):
        _ = named_array[3]

    with pytest.raises(KeyError):
        _ = named_array["d"]

    with pytest.raises(TypeError):
        _ = named_array[1.5]


def test_named_array_eq():
    # --- arrange -----------------------------------------
    named_array_1 = NamedArray(["a", "b", "c"], [1.2, 3.4, 5.6])
    named_array_2 = NamedArray(["a", "b", "c"], [1.2, 3.4, 5.6])

    # --- act & assert ------------------------------------
    assert named_array_1 == named_array_2
