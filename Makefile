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

test-without-numba:
	# run tests WITHOUT optional [benchmarking] dependencies installed
	uv sync;	# should remove numba from the environment
	uv run pytest ./tests

test-with-numba:
	# run tests WITH optional [numba] dependencies installed
	uv run --extra numba pytest ./tests

test: test-without-numba test-with-numba
	# run all tests

coverage:
	# run tests WITHOUT numba & create new report
	uv sync;	# should remove numba from the environment
	uv run --python 3.10 pytest ./tests --cov=./snuffled/ --cov-report=html
	# run tests WITH numba & append to report
	uv run --python 3.13 --extra numba pytest ./tests --cov=./snuffled/ --cov-append --cov-report=html

format:
	uvx ruff format .;
	uvx ruff check --fix .;

format-single-file:
	uvx ruff format ${file_path};
	uvx ruff check --fix ${file_path};
