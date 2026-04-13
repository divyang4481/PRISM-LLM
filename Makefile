.PHONY: install test lint format clean check

install:
	poetry install

test:
	poetry run pytest tests/ || true

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

check: lint test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info/
