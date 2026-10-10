"""Tests for the CI-matrix coverage check (scripts/check_python_matrix.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_python_matrix.py"


def _load_module():
    """Import the check by path: scripts/ is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("check_python_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_module()


def test_reads_only_quoted_matrix_python_values(tmp_path):
    """The parser picks up quoted `python:` matrix entries and ignores every `python_version:` key."""
    # --- arrange ----------------------
    workflow = tmp_path / "wf.yml"
    workflow.write_text(
        "        include:\n"
        '          - { python: "3.12", resolution: highest, jit: "off" }\n'
        '          - { python: "3.15.0rc1", resolution: highest, jit: "on" }\n'
        "          python_version: ${{ matrix.python }}\n"
        '          python_version: "3.13"\n',
        encoding="utf-8",
    )

    # --- act --------------------------
    tested = _mod.read_tested_versions(workflow)

    # --- assert -----------------------
    assert tested == {"3.12", "3.15.0rc1"}


@pytest.mark.parametrize(
    "declared, tested, expected_uncovered",
    [
        ({"3.12", "3.15"}, {"3.12", "3.15"}, set()),  # exact matches
        ({"3.15"}, {"3.15.0rc1"}, set()),  # a fuller pin of the same minor covers it
        ({"3.1"}, {"3.15"}, {"3.1"}),  # 3.15 is a different minor than 3.1
        ({"3.12", "3.15"}, {"3.12", "3.14"}, {"3.15"}),  # no entry for 3.15
    ],
)
def test_uncovered_versions(declared, tested, expected_uncovered):
    """A declared minor counts as covered by an exact entry or a fuller pin of that minor, and by nothing else."""
    # --- act --------------------------
    uncovered = _mod.uncovered_versions(declared, tested)

    # --- assert -----------------------
    assert uncovered == expected_uncovered


@pytest.mark.parametrize(
    "declared, tested, expected_exit_code",
    [
        ({"3.12", "3.15"}, {"3.12", "3.14"}, 1),  # 3.15 has no matrix entry
        ({"3.12"}, {"3.12", "3.14"}, 0),  # an extra matrix entry is fine
    ],
)
def test_main_fails_only_on_a_declared_version_without_a_matrix_entry(
    monkeypatch, declared, tested, expected_exit_code
):
    """The check fails on a declared version that has no matrix entry, and accepts extra matrix entries."""
    # --- arrange ----------------------
    monkeypatch.setattr(_mod, "read_declared_versions", lambda: declared)
    monkeypatch.setattr(_mod, "read_tested_versions", lambda: tested)

    # --- act --------------------------
    exit_code = _mod.main()

    # --- assert -----------------------
    assert exit_code == expected_exit_code


def test_repo_matrix_covers_declared_versions():
    """The repo's own `.python-versions` and CI test matrix satisfy the check."""
    # --- act / assert -----------------
    assert _mod.main() == 0
