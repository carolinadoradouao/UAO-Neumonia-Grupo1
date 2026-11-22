.PHONY: install run test check

install:
	uv sync

run:
	uv run python detector_neumonia.py

test:
	uv run pytest

check:
	uv run ruff check .
	uv run ruff format --check .
