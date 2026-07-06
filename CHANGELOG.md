# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- Package now ships type information (`py.typed`) for downstream type checkers
- PyPI classifiers (supported Python versions, typing, audience, topic)

### Changed
- Supported Python versions are now 3.11–3.14 (3.10 dropped, 3.14 added)
- Now requires numpy ≥ 2.0
- README badges and splash are now served from the repo / shields.io instead of gh-pages

### Deprecated

### Removed

### Fixed
- Fix broken splash screen CI job by committing a locally generated splash image

### Security

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
