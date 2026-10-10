file_path=
# PY sets the Python version for the local test targets.
PY ?= $(shell cat .python-version)

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help		                    Show this help message.'
	@echo ''
	@echo '  build		                    (Re)build package using uv.'
	@echo ''
	@echo '  dev-setup                      One-time: sync dev deps & install pre-commit hooks.'
	@echo '  test		                    Run the full pytest suite (JIT on) on Python $(PY).'
	@echo '  test-cov                       Local coverage estimate (Python $(PY), JIT off); approximates the CI coverage check, which combines coverage from every Python version.'
	@echo '  lint		                    Run all pre-commit hooks on all files.'
	@echo '  format		                    Format source code using ruff.'
	@echo '  format-single-file             Format single file using ruff. Useful in e.g. PyCharm to automatically trigger formatting on file save.'
	@echo ''
	@echo '  splash       			        Build splash screen using current version of package.'
	@echo ''
	@echo '  release       		            Release a version: make release VERSION=X.Y.Z (validates, stamps, tags, pushes).'
	@echo ''
	@echo 'Options:'
	@echo ''
	@echo '  test, test-cov                 - accept `PY=<x.y>` to run on another Python than the default in .python-version.'
	@echo '  format-single-file             - accepts `file_path=<path>` to pass the relative path of the file to be formatted.'

build:
	uv build;

dev-setup:
	uv sync
	uv run pre-commit install

test:
	# full suite, JIT on - just 1 python version
	uv run --python $(PY) pytest ./tests

test-cov:
	# This target is a local coverage estimate: it runs with JIT off (so coverage sees inside
	# @njit bodies) on one Python only.
	# Not the CI coverage check — that combines coverage from every Python version, so
	# version-divergent lines may read as uncovered here. Use it as a cheap "did I keep
	# coverage up" check before pushing.
	NUMBA_DISABLE_JIT=1 uv run --python $(PY) pytest ./tests --cov --cov-report=term-missing

lint:
	uv run pre-commit run --all-files

format:
	uv run ruff format .;
	uv run ruff check --fix .;

format-single-file:
	uv run ruff format ${file_path};
	uv run ruff check --fix ${file_path};

splash:
	./.github/scripts/create_splash.sh "$$(uv version --short)-dev";

release:
	@test -n "$(VERSION)" || (echo "Usage: make release VERSION=X.Y.Z" && exit 1)
	$(MAKE) test
	uv run python scripts/release.py $(VERSION)
