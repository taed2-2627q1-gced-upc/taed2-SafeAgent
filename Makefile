.DEFAULT_GOAL := help

.PHONY: help requirements fetch data pipeline train evaluate experiment test coverage lint docs

help:
	@echo "Targets: requirements fetch data pipeline train evaluate experiment test lint docs"

## Install the locked environment
requirements:
	uv sync --frozen --group data --group dev --group training

## Download and check the pinned source
fetch:
	uv run --frozen --group data python -m taed2_safeagent.dataset fetch

## Reproduce the data pipeline
data:
	uv run --frozen --group data dvc repro validate

## Reproduce data and model results
pipeline:
	uv run --frozen --group data dvc repro

## Train the first model
train:
	uv run --frozen --group data dvc repro train

## Evaluate on validation data
evaluate:
	uv run --frozen --group data dvc repro evaluate

## Record a shared baseline run
experiment:
	uv run --frozen --group data python -m taed2_safeagent.modeling.experiment --params params.yaml

## Check data and model behavior
test:
	uv run --frozen --group data pytest -q

## Measure test coverage on source code
coverage:
	uv run --frozen --group data --group dev pytest --cov=taed2_safeagent --cov-report=term-missing --cov-report=xml tests/

## Check changed Python code
lint:
	uv run --frozen --group data pylint taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling taed2_safeagent/api tests kaggle_train.py
	uv run --frozen --group data ruff check taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling taed2_safeagent/api tests kaggle_train.py
	uv run --frozen --group data ruff format --check taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling taed2_safeagent/api tests kaggle_train.py

## Build public docs
docs:
	uv run --frozen --group data mkdocs build --strict
