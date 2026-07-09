file_path=

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help		                    Show this help message.'
	@echo ''
	@echo '  build		                    (Re)build package using uv.'
	@echo ''
	@echo '  dev-setup                      One-time: sync dev deps & install pre-commit hooks.'
	@echo '  test		                    Run the full pytest suite (JIT on).'
	@echo '  test-cov                       Run the coverage-bearing tier with JIT disabled + per-line coverage report.'
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
	@echo '  format-single-file             - accepts `file_path=<path>` to pass the relative path of the file to be formatted.'

build:
	uv build;

dev-setup:
	uv sync
	uv run pre-commit install

test:
	# full suite (both test tiers), JIT on - just 1 python version
	uv run --python 3.13 pytest ./tests

test-cov:
	# coverage-bearing tier only: JIT disabled so per-line coverage sees inside @njit bodies
	NUMBA_DISABLE_JIT=1 uv run --python 3.13 pytest ./tests --cov --cov-report=term-missing

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
