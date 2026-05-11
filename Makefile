.PHONY: install seed dbt-deps api web dev test fmt lint

install:
	uv sync --extra dev
	cd web && pnpm install

seed:
	uv run python scripts/seed_synthetic.py

dbt-deps:
	uv run dbt deps --project-dir dbt/automl_dbt

api:
	uv run uvicorn api.main:app --reload --port 8000

web:
	cd web && pnpm dev

dev:
	bash scripts/dev.sh

test:
	uv run pytest -q

fmt:
	uv run ruff format .

lint:
	uv run ruff check . && uv run mypy core agents api cli synthetic
