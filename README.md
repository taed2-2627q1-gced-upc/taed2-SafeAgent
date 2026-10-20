# SafeAgent

SafeAgent classifies proposed shell commands as ALLOW, ASK or DENY.
The project compares command inputs with command and execution context inputs.
The component reads commands as text and never runs them.

## Reproduce the project

Use Python 3.11 and uv. Start with the [setup guide](docs/getting-started.md).
It explains how to connect DVC and set up the local cache.

```sh
uv sync --frozen --group data --group dev --group training
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
The [encoder guide](docs/encoder-training.md) explains shared training for
CodeBERT and ModernBERT with both input modes.
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

## Command API

The [API guide](docs/api.md) explains how to serve the saved baseline through
FastAPI, validate one command per request and run the HTTP tests. The
[deployment guide](docs/deployment.md) documents running the service on the
Virtech private cloud.

## Energy consumption and emissions

Training and inference energy are tracked with [CodeCarbon](https://codecarbon.io/)
to quantify computational resource usage and environmental impact across the baseline
and candidate encoder architectures. CodeCarbon kilogram readings are converted to grams
(g CO2eq) for readability:

### Training energy by model

| Model | Input mode | Hardware | Duration | Energy (kWh) | Emissions (g CO2eq) | MLflow run |
| --- | --- | --- | --- | --- | --- | --- |
| **TF-IDF + Linear SVM** | `command` | CPU (WSL2) | 6.68s | 0.0001 kWh | 0.036 g | [`80c83cb3`](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/80c83cb3105a428db83b5038ab37c0ec) |
| **CodeBERT-base** | `command` | 2 x Tesla T4 | 478.78s | 0.0156 kWh | 5.780 g | [`a81f878b`](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/a81f878bbbff4019b032b389a0909c01) |
| **CodeBERT-base** | `command-context` | 2 x Tesla T4 | 834.08s | 0.0273 kWh | 10.073 g | [`c3532eb8`](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/c3532eb8a9c34146926bc9d2e10d047a) |
| **ModernBERT-base** | `command` | 2 x Tesla T4 | 791.29s | 0.0259 kWh | 9.555 g | [`4851c0df`](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/4851c0df192344fa99eacf1721bdbafe) |
| **ModernBERT-base** | `command-context` | 2 x Tesla T4 | 1353.73s | 0.0443 kWh | 16.380 g | [`5ed5c367`](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/5ed5c367a6c8481d93a744ce97e7a7b0) |

### CPU inference energy (1 vCPU benchmark)

Inference efficiency was evaluated on a controlled validation batch under single-thread
CPU execution to model the Virtech deployment environment:

| Model | Input mode | Latency (p50) | Energy / query | Emissions / query | Scope / Status |
| --- | --- | --- | --- | --- | --- |
| **TF-IDF + Linear SVM** | `command` | **0.45 ms** | **0.000004 Wh** | **0.0016 mg** | Measured (500 queries) |
| **CodeBERT-base** | `command` | 831.00 ms | 0.0148 Wh | 5.642 mg | D4 latency estimate |
| **CodeBERT-base** | `command-context` | 12,639.40 ms | 0.2249 Wh | 85.458 mg | D4 latency estimate |
| **ModernBERT-base** | `command` | 5,390.19 ms | 0.0964 Wh | 36.617 mg | D4 latency estimate |
| **ModernBERT-base** | `command-context` | 616.88 ms | 0.0070 Wh | 2.597 mg | Measured (controlled batch) |

The linear SVM baseline demonstrates orders-of-magnitude lower energy consumption and
latency per query on CPU, making it the most energy-efficient choice for real-time edge
or CPU cloud serving. Detailed metrics are preserved in `reports/inference/energy.json`.

## Project layout

| Folder or file | Contents |
| --- | --- |
| taed2_safeagent/data | Source checks, audit, groups, splits, validation and input builder |
| taed2_safeagent/api | Command schemas, saved model loading and HTTP routes |
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

## Team workflow

The [team workflow guide](docs/team-workflow.md) links the Trello board to
branches and pull requests, includes a dated board export, and explains Slack
coordination and the GitHub notification setup. See
[feature ownership](docs/feature-ownership.md) for maintenance and review roles.

## Quality checks

```sh
uv run --frozen --group data pylint taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling tests
uv run --frozen --group data pytest -q
uv run --frozen --group data mkdocs build --strict
```

GitHub Actions runs data and small model tests in one job. It downloads the
public pinned snapshot and does not need personal DagsHub credentials.
PyNBLint is installed for notebook checks when notebooks are available. Tracking tests
use a fake client, so CI needs no DagsHub credentials. Encoder training also
records CodeCarbon readings, with estimates and missing sensors reported.

Use make help for optional convenience commands. The uv commands also work
without make.
