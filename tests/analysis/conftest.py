from typing import Callable

import pytest


@pytest.fixture
def test_fun_quad() -> Callable[[float], float]:
    """
    A simple quadratic function for testing purposes, with a root at x = sqrt(0.5).
    """
    return lambda x: x**2 - 0.5
