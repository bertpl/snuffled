# Contributing to snuffled

Thanks for your interest in contributing.

## Dev setup

One-time setup on a fresh clone:

```bash
make dev-setup
```

This syncs dev dependencies via `uv` and installs the pre-commit hooks.

## Common commands

```bash
make test    # Run the test suite (pytest)
make format  # Format and auto-fix with ruff
make lint    # Run all pre-commit hooks (ruff, ty, file hygiene, ...) over all files
```

## Branching

Branch names follow the pattern:

```
<prefix>/<short-slug>
```

- **Prefix** — one of `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.
  CI rejects anything else.
- **Slug** — short kebab-case description: lowercase letters, digits, and
  hyphens only.

Examples: `feat/yaml-input`, `fix/root-width-crash`, `chore/bump-numpy`.

## Pull requests

PRs are merged into `main` via **squash merge only** (repo settings disable
merge commits and rebase merges). Each PR therefore produces exactly one commit
on `main`. The squash commit subject is the PR title, so write it with care — it
becomes the permanent history, and CI checks it against the `<prefix>: <summary>`
convention below. The feature branch is deleted automatically on merge.

## Commit messages

Subject line uses the same short-form prefixes as branches:

```
<prefix>: <imperative summary>
```

- **Prefix** — `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.
- **Summary** — imperative mood, lowercase, no trailing period.

Examples:

```
feat: add yaml input parser
fix: handle empty root interval
chore: bump numpy floor
```

## Changelog

Add an entry under the appropriate category in the `## Unreleased` section of
[`CHANGELOG.md`](CHANGELOG.md) as part of your PR. CI requires this for `feat/`
and `fix/` branches.

Changelog entries are **user-facing** — write them for someone deciding whether
to upgrade, not for someone reviewing the implementation. Focus on what changed
from the user's perspective.

**Keep each entry to a single line.** Avoid verbosity; omit internal details
(class names, wiring, refactors that don't affect behavior). Expand to a second
line only when a single line genuinely can't convey what the change is about.
