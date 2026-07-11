"""Test-suite configuration — and the canonical explanation of why it runs in two modes.

numba compiles `@njit` functions to machine code, which `coverage.py` cannot trace
per line: on a normal (JIT-on) run every compiled kernel body reads as uncovered. So
the suite is exercised in two complementary modes:

- **JIT off** (`NUMBA_DISABLE_JIT=1`): decoration becomes a no-op and the kernels run as
  plain Python, so `coverage.py` sees inside them. This is the mode that measures
  coverage — CI collects it here and unions the per-Python results into the gate.
- **JIT on** (default): the real compiled paths run, at production speed. This is the
  only mode that can afford the slow whole-pipeline tests, and the only mode where
  compiled-code behavior is meaningful.

Tests that only make sense with JIT on — slow whole-pipeline sweeps, or assertions about
compiled behavior — carry the `only_with_numba_jit` marker (an importable alias in
`tests/helpers.py`, so a typo fails at collection under `--strict-markers`). The hook
below auto-skips them whenever JIT is disabled, so the coverage mode never runs them.
"""

import numba
import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `only_with_numba_jit` tests when numba runs with JIT disabled.

    Keys on `numba.config.DISABLE_JIT` — numba's own parse of the `NUMBA_DISABLE_JIT`
    environment variable — so the skip decision always matches what numba actually does.
    """
    if not numba.config.DISABLE_JIT:
        return
    skip = pytest.mark.skip(reason="requires numba JIT (running with NUMBA_DISABLE_JIT)")
    for item in items:
        if "only_with_numba_jit" in item.keywords:
            item.add_marker(skip)
