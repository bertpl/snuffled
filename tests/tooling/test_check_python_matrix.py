"""Test the CI-matrix coverage check in scripts/check_python_matrix.py."""

import pytest

from .helpers import import_script

check_python_matrix = import_script("check_python_matrix")


def test_reads_only_quoted_matrix_python_values(tmp_path):
    """`read_tested_versions` picks up quoted `python:` matrix entries and ignores every `python_version:` key."""
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
    tested_versions = check_python_matrix.read_tested_versions(workflow)

    # --- assert -----------------------
    assert tested_versions == {"3.12", "3.15.0rc1"}


@pytest.mark.parametrize(
    "declared_versions, tested_versions, expected_uncovered",
    [
        ({"3.12", "3.15"}, {"3.12", "3.15"}, set()),
        ({"3.15"}, {"3.15.0rc1"}, set()),  # a 3.15.0rc1 entry covers the declared minor version 3.15
        ({"3.1"}, {"3.15"}, {"3.1"}),  # a 3.15 entry does not cover minor version 3.1
        ({"3.12", "3.15"}, {"3.12", "3.14"}, {"3.15"}),
    ],
)
def test_find_uncovered_versions(declared_versions, tested_versions, expected_uncovered):
    """A declared minor version counts as covered by an entry with the same version, or by an entry whose version
    starts with it followed by a dot."""
    # --- act --------------------------
    uncovered = check_python_matrix.find_uncovered_versions(declared_versions, tested_versions)

    # --- assert -----------------------
    assert uncovered == expected_uncovered


@pytest.mark.parametrize(
    "declared_versions, tested_versions, expected_exit_code",
    [
        ({"3.12", "3.15"}, {"3.12", "3.14"}, 1),
        ({"3.12"}, {"3.12", "3.14"}, 0),  # an extra matrix entry is fine
    ],
)
def test_main_fails_only_on_a_declared_version_without_a_matrix_entry(
    monkeypatch, declared_versions, tested_versions, expected_exit_code
):
    """The check fails on a declared version that has no matrix entry, and accepts extra matrix entries."""
    # --- arrange ----------------------
    monkeypatch.setattr(check_python_matrix, "read_declared_versions", lambda: declared_versions)
    monkeypatch.setattr(check_python_matrix, "read_tested_versions", lambda: tested_versions)

    # --- act --------------------------
    exit_code = check_python_matrix.main()

    # --- assert -----------------------
    assert exit_code == expected_exit_code


def test_repo_matrix_covers_declared_versions():
    """The repo's own `.python-versions` and CI test matrix satisfy the check."""
    # --- act / assert -----------------
    assert check_python_matrix.main() == 0
