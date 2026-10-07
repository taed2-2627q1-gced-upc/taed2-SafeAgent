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
The [model cards](docs/model-cards/index.md) describe the classifiers and their limits.

## Train the first model

The first baseline uses character TF-IDF and a linear SVM with command inputs.
After preparing the data, run:

```sh
uv run --frozen --group data python -m taed2_safeagent.modeling.train --params params.yaml
uv run --frozen --group data python -m taed2_safeagent.modeling.evaluate --params params.yaml
```

See the [baseline guide](docs/baseline.md) for recovery and output files.
The [validation results](reports/baseline/metrics.json) give macro F1 0.8613
and DENY recall 0.8613, with 19 DENY examples predicted as ALLOW.
The test partition is reserved for the final model comparison.

## Project layout

| Folder or file | Contents |
| --- | --- |
| taed2_safeagent/data | Source checks, audit, groups, splits, validation and input builder |
| taed2_safeagent/dataset.py | Data command line interface |
| taed2_safeagent/features.py | Character features for the baseline |
| taed2_safeagent/modeling | Training, model loading and validation evaluation |
| data/raw/shell_safety | Immutable source snapshot stored with DVC |
| data/interim/shell_safety | Clean inputs, provenance and quarantine stored with DVC |
| data/processed/shell_safety | Prepared splits and manifest stored with DVC |
| reports/data | Small deterministic audit, split and validation reports |
| models/tfidf_char_linear_svm | Fitted pipeline and metadata stored with DVC |
| reports/baseline | Validation metrics and predictions with prepared IDs |
| tests | Data and model behavior tests |
| docs | Public setup guide, Dataset Card and model cards |
| params.yaml | Fixed data protocol and baseline settings |
| dvc.yaml and dvc.lock | Pipeline and artifact versions |
| pyproject.toml and uv.lock | Dependencies and locked environment |

The plotting module is still a starter file. The project started from
Cookiecutter Data Science.

## Quality checks

```sh
uv run --frozen --group data pylint taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling tests
uv run --frozen --group data pytest -q
uv run --frozen --group data mkdocs build --strict
```

GitHub Actions runs data and small model tests in one job. It downloads the
public pinned snapshot and does not need personal DagsHub credentials.
PyNBLint is installed, but there are no notebooks to check yet. Shared MLflow
tracking is planned for the next PR and CodeCarbon measurements
for Milestone 3.

Use make help for optional convenience commands. The uv commands also work
without make.
