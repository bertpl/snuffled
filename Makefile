file_path=

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help		                    Show this help message.'
	@echo ''
	@echo '  build		                    (Re)build package using uv.'
	@echo ''
	@echo '  test		                    Run pytest unit tests.'
	@echo '  format		                    Format source code using ruff.'
	@echo '  format-single-file             Format single file using ruff. Useful in e.g. PyCharm to automatically trigger formatting on file save.'
	@echo ''
	@echo 'Options:'
	@echo ''
	@echo '  format-single-file             - accepts `file_path=<path>` to pass the relative path of the file to be formatted.'

build:
	uv build;

test-with-numba:
	# run tests WITH optional [numba] dependencies installed
	uv run --extra numba pytest ./tests

test-without-numba:
	# run tests WITHOUT optional [benchmarking] dependencies installed
	uv sync;	# should remove numba from the environment
	uv run pytest ./tests

test: test-with-numba test-without-numba
	# run all tests

coverage:
	# run tests WITH numba & create new report
	uv run --extra numba pytest ./tests --cov=./snuffled/ --cov-report=html
	# run tests WITHOUT numba & append to report
	uv sync;	# should remove numba from the environment
	uv run pytest ./tests --cov=./snuffled/ --cov-append --cov-report=html

format:
	uvx ruff format .;
	uvx ruff check --fix .;

format-single-file:
	uvx ruff format ${file_path};
	uvx ruff check --fix ${file_path};
