all: run

install:
	uv sync

run:
	un run python src/main.py

debug:
	un run python -m pdb src/main.py

clean:
	rm -rf __pycache__ .mypy_cache

lint:
	flake8 .
	mypy . \
	--warn-return-any \ 
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

.PHONY: all install run debug clean lint lint-strict