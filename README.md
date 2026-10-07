# SafeAgent

SafeAgent classifies proposed shell commands as ALLOW, ASK or DENY.
We compare command inputs with command and execution context inputs.
The component reads commands as text and never runs them.

## Reproduce the data

Use Python 3.11 and uv. Start with the [setup guide](docs/getting-started.md).
It explains how to connect DVC and set up the local cache.

```sh
uv sync --frozen --group data --group dev
uv run --frozen --group data dvc pull
uv run --frozen --group data dvc repro
uv run --frozen --group data pytest -q
```

The [Dataset Card](docs/dataset-card.md) explains the pinned source, cleaning,
group split, input formats, checks and known limits.
The [model cards](docs/model-cards/index.md) describe the planned classifiers.
Training results are pending.

## Project layout

| Folder or file | Contents |
| --- | --- |
| taed2_safeagent/data | Source checks, audit, groups, splits, validation and input builder |
| taed2_safeagent/dataset.py | Data command line interface |
| data/raw/shell_safety | Immutable source snapshot stored with DVC |
| data/interim/shell_safety | Clean inputs, provenance and quarantine stored with DVC |
| data/processed/shell_safety | Prepared splits and manifest stored with DVC |
| reports/data | Small deterministic audit, split and validation reports |
| tests | Data behavior and integrity tests |
| docs | Public setup guide, Dataset Card and model cards |
| params.yaml | Source hashes and fixed data protocol |
| dvc.yaml and dvc.lock | Pipeline and artifact versions |
| pyproject.toml and uv.lock | Dependencies and locked environment |

The remaining model and plotting modules are starter files. They do not train
or evaluate a model yet. The project started from Cookiecutter Data Science.

## Quality checks

```sh
uv run --frozen --group data pylint taed2_safeagent/data taed2_safeagent/dataset.py tests
uv run --frozen --group data pytest -q
uv run --frozen --group data mkdocs build --strict
```

GitHub Actions runs the data checks. It downloads the
public pinned snapshot and does not need personal DagsHub credentials.
PyNBLint is installed, but there are no notebooks to check yet. MLflow and
CodeCarbon measurements will be added with real model training.

Use make help for optional convenience commands. The uv commands also work
without make.
