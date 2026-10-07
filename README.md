# SafeAgent

SafeAgent classifies proposed shell commands as ALLOW, ASK or DENY.
The project compares command inputs with command and execution context inputs.
The component reads commands as text and never runs them.

## Reproduce the project

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
The DVC pipeline checks the data, trains the model and evaluates validation.
To run these stages separately:

```sh
uv run --frozen --group data dvc repro train
uv run --frozen --group data dvc repro evaluate
```

See the [baseline guide](docs/baseline.md) for recovery and output files.
The [validation results](reports/baseline/metrics.json) give macro F1 0.8613
and DENY recall 0.8613, with 19 DENY examples predicted as ALLOW.
The test partition is reserved for the final model comparison.

The baseline also supports shared MLflow tracking. Follow the
[experiment setup](docs/baseline.md#record-an-experiment) to configure credentials,
then run from a clean committed checkout:

```sh
uv run --frozen --group data python -m taed2_safeagent.modeling.experiment --params params.yaml
```

This trains the baseline, evaluates validation, uploads the model with DVC and
records the run in DagsHub. The run link is saved in reports/baseline/mlflow_run.json.
Normal DVC reproduction reuses current outputs and does not create an MLflow run.
Use dvc repro validate when only the data is needed.

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

The project started from Cookiecutter Data Science.

## Quality checks

```sh
uv run --frozen --group data pylint taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling tests
uv run --frozen --group data pytest -q
uv run --frozen --group data mkdocs build --strict
```

GitHub Actions runs data and small model tests in one job. It downloads the
public pinned snapshot and does not need personal DagsHub credentials.
PyNBLint is installed for notebook checks when notebooks are available. Tracking tests
use a fake client, so CI needs no DagsHub credentials. CodeCarbon measurements
remain planned for Milestone 3.

Use make help for optional convenience commands. The uv commands also work
without make.
