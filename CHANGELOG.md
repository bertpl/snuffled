# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## 0.1.8 (2026-07-12)

### Added
- `ALL_ROOTS_TOO_CLOSE_TO_EDGE` diagnostic — flags functions whose every root is too close to an interval edge to analyze reliably
- `NAN_VALUES_DETECTED` and `INF_VALUES_DETECTED` diagnostics — flag functions that return NaN or ±inf for some input

### Changed
- Sample-count parameters (`n_fun_samples`, `n_roots`, `n_root_samples`) below 10 now raise a clear `ValueError`

### Fixed
- `extract_all()` no longer crashes on functions with no root in the interval (e.g. `1 + x²`)
- `extract_all()` no longer crashes on flat / constant functions
- `extract_all()` no longer crashes on roots very close to an interval edge (such roots are excluded from analysis)
- `extract_all()` no longer crashes on NaN-returning functions; pole / ±inf functions now produce finite scores (previously NaN)
- `extract_all()` no longer crashes on all-zeros / zero-magnitude functions
## 0.1.7 (2026-07-11)

### Changed
- **breaking:** numba is now a required dependency (was the optional `snuffled[numba]` extra); pure-Python execution remains available for development via `NUMBA_DISABLE_JIT=1`
## 0.1.6 (2026-07-06)

### Security
- Releases now ship SLSA build provenance and a GitHub Release with the changelog excerpt
## 0.1.5 (2026-07-06)

### Added
- Package now ships type information (`py.typed`) for downstream type checkers
- PyPI classifiers (supported Python versions, typing, audience, topic)

### Changed
- Supported Python versions are now 3.11–3.14 (3.10 dropped, 3.14 added)
- Now requires numpy ≥ 2.0
- README badges and splash are now served from the repo / shields.io instead of gh-pages

### Fixed
- Fix broken splash screen CI job by committing a locally generated splash image
## 0.1.4 (2025-11-07)

### Changed
- Improve CI/CD pipeline for efficiency
- Improve splash screen setup
- Move to trunk-based development workflow with release branches

## 0.1.3 (2025-10-14)

### Changed
- Improve CI/CD pipeline for efficiency
- Improve splash screen setup

## 0.1.2 (2025-10-11)

### Added
- Add Change Log

### Changed
- Minor tweaks to CI/CD pipeline

## 0.1.1 (2025-09-06)

### Changed
- Minor tweaks to CI/CD pipeline

## 0.1.0 (2025-09-06)

### Added
- Initial feature-complete version
- Full test suite & automatic badge generation for README.md

### Changed
- Initial CI/CD pipeline
