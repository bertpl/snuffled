"""Test the changelog finalizer in scripts/release.py."""

from datetime import date

import pytest

from .helpers import load_script

release = load_script("release")


@pytest.mark.parametrize(
    "previous_release, separator",
    [
        ("## 0.1.0 (2025-01-01)\n\n### Added\n- Initial version\n", "\n"),
        ("", ""),  # first release: the finalized section ends the file
    ],
    ids=["previous-release", "first-release"],
)
def test_finalize_changelog_separates_sections(tmp_path, monkeypatch, previous_release, separator):
    """The finalized section ends in a blank line before the previous release's heading, and in 1 newline at
    end-of-file."""
    # --- arrange ----------------------
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## Unreleased\n\n### Added\n- Add a feature\n\n### Fixed\n\n" + previous_release
    )
    monkeypatch.setattr(release, "CHANGELOG", changelog)

    # --- act --------------------------
    release.step_10_finalize_changelog("1.2.3")

    # --- assert -----------------------
    expected_section = f"## 1.2.3 ({date.today().isoformat()})\n\n### Added\n- Add a feature\n"
    assert changelog.read_text() == "# Changelog\n\n" + expected_section + separator + previous_release
