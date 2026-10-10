"""Test that the installed package ships the files that its classifiers promise."""

from pathlib import Path

import snuffled


def test_package_ships_py_typed_marker():
    """The package ships `py.typed`, without which type checkers ignore its annotations."""
    # --- act / assert -----------------
    assert (Path(snuffled.__file__).parent / "py.typed").is_file()
