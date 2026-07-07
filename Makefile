VENV = .venv
BIN = $(VENV)/bin

all: run

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

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
	uv run ypy . --strict

.PHONY: all install run debug clean lint lint-strict format