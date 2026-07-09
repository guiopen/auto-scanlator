format:
	uv run ruff format src/ main.py

lint:
	uv run ruff check src/ main.py

typecheck:
	uv run mypy src/ main.py

check: format lint typecheck