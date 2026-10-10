"""Check that the CI test matrix covers every Python version listed in `.python-versions`.

`.python-versions` lists the package's supported Python versions. `scripts/release.py` checks that list against the
trove classifiers, but nothing checks it against CI: the test matrix in `.github/workflows/_unit_tests.yml` is edited
by hand.

A version can therefore be listed in `.python-versions`, pass the classifier check in `scripts/release.py`, and be
released to PyPI with no CI job testing it.

Usage:

    python scripts/check_python_matrix.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_VERSIONS_FILE = REPO_ROOT / ".python-versions"
UNIT_TESTS_WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "_unit_tests.yml"

# Matrix entries give their Python version as `python: "<version>"`. The pattern requires the colon directly after
# `python`, so the `python_version:` input that the workflow passes to its `unit_test` action never matches, even
# with a quoted value.
_MATRIX_PYTHON_RE = re.compile(r'\bpython:\s*"([^"]+)"')


def read_declared_versions(path: Path = PYTHON_VERSIONS_FILE) -> set[str]:
    """Return the Python minor versions declared in `.python-versions` (one per non-empty line)."""
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_tested_versions(path: Path = UNIT_TESTS_WORKFLOW_FILE) -> set[str]:
    """Return the Python version of every entry in the workflow's test matrix."""
    return set(_MATRIX_PYTHON_RE.findall(path.read_text(encoding="utf-8")))


def find_uncovered_versions(declared_versions: set[str], tested_versions: set[str]) -> set[str]:
    """Return the declared versions that no matrix entry covers; a matrix entry for an undeclared version is ignored.

    A declared minor version is covered by an entry with the same version, or by an entry whose version starts with
    that minor version followed by a dot (`3.15` is covered by `3.15.0rc1`). The dot keeps a `3.15` entry from
    covering `3.1`.
    """
    uncovered = set()
    for version in declared_versions:
        is_covered = False
        for entry in tested_versions:
            if entry == version or entry.startswith(f"{version}."):
                is_covered = True
        if not is_covered:
            uncovered.add(version)
    return uncovered


def main() -> int:
    """Report any declared Python version that the CI matrix does not test; return 1 if any."""
    declared_versions = read_declared_versions()
    tested_versions = read_tested_versions()
    uncovered = find_uncovered_versions(declared_versions, tested_versions)
    if uncovered:
        print(
            f".python-versions declares {sorted(uncovered)} with no matching entry in the "
            f"{UNIT_TESTS_WORKFLOW_FILE.relative_to(REPO_ROOT)} test matrix "
            f"(the matrix has entries for {sorted(tested_versions)}).",
            file=sys.stderr,
        )
        return 1
    else:
        print(f"CI matrix covers all {len(declared_versions)} declared Python versions: {sorted(declared_versions)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
