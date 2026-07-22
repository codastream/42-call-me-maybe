VENV = .venv
BIN = $(VENV)/bin
PYTHONPATH_ENV = PYTHONPATH=.

all: run

install:
	uv sync

run:
	uv run python -m src

run-robust:
	uv run python -m src -input 'data/input/function_calling_tests_robustness.json'

debug:
	uv run python -m pdb -m src

test:
	$(PYTHONPATH_ENV) uv run python src/tests/test_decoding.py

test-v:
	$(PYTHONPATH_ENV) uv run python src/tests/test_decoding.py -v

test-debug:
	$(PYTHONPATH_ENV) uv run python src/tests/test_decoding.py --log-level DEBUG

clean:
	rm -rf __pycache__ .mypy_cache

format:
	$(BIN)/autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive .
	$(BIN)/autopep8 --in-place --recursive --global-config .flake8 .

lint:
	uv run flake8 . --statistics
	uv run mypy .

lint-strict:
	uv run flake8 .
	uv run python -m mypy src --explicit-package-bases --strict

.PHONY: all install run debug clean lint lint-strict format