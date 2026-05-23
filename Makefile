.PHONY: install test lint format clean

install:
	pip install -e .
	pip install -r requirements.txt

test:
	python -m pytest

lint:
	ruff check .

format:
	ruff format . --check

format-fix:
	ruff format .

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
