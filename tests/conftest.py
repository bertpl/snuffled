import numba
import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tier-B tests (`only_with_numba_jit`) when numba runs with JIT disabled.

    Keys on `numba.config.DISABLE_JIT` — numba's own parse of the `NUMBA_DISABLE_JIT`
    environment variable — so the skip decision always matches what numba actually does.
    """
    if not numba.config.DISABLE_JIT:
        return
    skip = pytest.mark.skip(reason="requires numba JIT (running with NUMBA_DISABLE_JIT)")
    for item in items:
        if "only_with_numba_jit" in item.keywords:
            item.add_marker(skip)
