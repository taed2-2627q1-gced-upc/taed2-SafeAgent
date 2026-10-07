.DEFAULT_GOAL := help

.PHONY: help requirements fetch data test lint docs

help:
	@echo "Targets: requirements fetch data test lint docs"

## Install the locked environment
requirements:
	uv sync --frozen --group data --group dev

## Download and check the pinned source
fetch:
	uv run --frozen --group data python -m taed2_safeagent.dataset fetch

## Reproduce the data pipeline
data:
	uv run --frozen --group data dvc repro

## Check data behavior
test:
	uv run --frozen --group data pytest -q

## Check changed Python code
lint:
	uv run --frozen --group data pylint taed2_safeagent/data taed2_safeagent/dataset.py tests
	uv run --frozen --group data ruff check taed2_safeagent/data taed2_safeagent/dataset.py tests
	uv run --frozen --group data ruff format --check taed2_safeagent/data taed2_safeagent/dataset.py tests

## Build public docs
docs:
	uv run --frozen --group data mkdocs build --strict
