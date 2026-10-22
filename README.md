<p align="center">
  <br>
  <img src="docs/assets/safeagent-banner.png" alt="SafeAgent" width="280">
  <br>
</p>

<p align="center">
SafeAgent checks shell commands proposed by an AI coding agent and returns<br>
<strong>ALLOW</strong>, <strong>ASK</strong> or <strong>DENY</strong>. It reads the command as text and never executes it.<br>
The project compares models that use only the command with models that also
use the session context.
</p>

<p align="center">
<a href="http://nattech.fib.upc.edu:40490/docs">Try the live API</a> ·
<a href="http://nattech.fib.upc.edu:40490/openapi.json">OpenAPI schema</a> ·
<a href="docs/dataset-card.md">Dataset Card</a> ·
<a href="docs/model-cards/index.md">Model cards</a> ·
<a href="https://dagshub.com/Pau-Balaguer/taed2-SafeAgent">DagsHub and experiments</a>
</p>

## Try the API

The [Swagger page](http://nattech.fib.upc.edu:40490/docs) runs against the API
deployed on Virtech. It can be used directly in a browser without installing
the project.

First, open GET /ready and click Try it out, then Execute. This shows whether
the model is loaded and which version is running. Next, open POST /predict
and choose one of the examples.

* `git status` returns ALLOW.
* `git branch feature` returns ALLOW.
* `git branch -d feature` returns ASK.
* `rm -rf /` returns DENY.
* An empty command returns HTTP 422.

These are examples checked with the deployed SVM. A prediction helps the calling
agent decide what to do, while the command itself stays as text.

The [API guide](docs/api.md) explains the request fields, responses and errors.
The [demo guide](docs/demo.md) gives the short presentation walkthrough.

## Set up the project

### Install the tools and get the code

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The project uses Python 3.11, which uv can install.

```sh
git clone https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent.git
cd taed2-SafeAgent
uv python install 3.11
uv sync --frozen --group data --group dev
```

Run the following commands from this repository folder. uv creates the local
environment and installs the versions saved in uv.lock, so there is no need
to activate the environment before each command.

This setup is enough for the API and the data checks. The training libraries
are added later when reproducing the experiments.

### Connect to the model storage

Code and DVC pointers are stored on GitHub. Data and model files are stored
on the shared [DagsHub project](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent).

Open its storage settings and get the personal S3 access key and secret.
Replace the placeholders below with those values.

```sh
uv run --frozen dvc remote modify --local storage access_key_id YOUR_ACCESS_KEY
uv run --frozen dvc remote modify --local storage secret_access_key YOUR_SECRET_KEY
```

This saves the credentials in .dvc/config.local, which Git ignores. Keep the
values private. The remote address is already configured in the repository.
The storage also accepts AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY when
they are already set in the environment.

On Windows, a long checkout path can cause DVC cache errors. The
[setup guide](docs/getting-started.md#windows-cache) explains how to use a short
cache path.

### Run the API locally

Download just the SVM used by the deployed service, then start FastAPI.

```sh
uv run --frozen dvc pull models/vm_linear_svm
uv run --frozen uvicorn taed2_safeagent.api.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs and try the same examples. Leave that terminal
open while using the API. Press Ctrl+C to stop it.

In another terminal, check the running service with:

```sh
uv run --frozen python scripts/smoke_test.py --url http://127.0.0.1:8000
```

The check sends valid and invalid requests over HTTP. A failure returns a
nonzero exit code. To check the deployed demo and its documentation instead:

```sh
uv run --frozen python scripts/smoke_test.py --demo --url http://nattech.fib.upc.edu:40490
```

The VM uses models/vm_linear_svm. SAFEAGENT_MODEL_DIR can select another
compatible saved SVM. If the selected model cannot load, /health stays available
while /ready and /predict return HTTP 503. The
[deployment guide](docs/deployment.md) covers VM setup, public access and
the scripts for starting, stopping and restarting the service.

## Reproduce the project

The full environment adds PyTorch, Transformers and CodeCarbon. It takes more
space than the API setup.

```sh
uv sync --frozen --group data --group dev --group training
uv run --frozen --group data dvc pull
uv run --frozen --group data --group training dvc repro
uv run --frozen --group data --group training pytest -q
```

A full DVC pull can also download the encoder weights. The API only needs the
smaller model downloaded in the previous section.

The pipeline checks the source, prepares the grouped splits, validates the data,
trains the baseline and evaluates it on validation. It also has a separate
train_vm stage for the deployment model. DVC reuses outputs when their recorded
inputs have not changed.

For data preparation alone, use:

```sh
uv run --frozen --group data dvc repro validate
```

If DagsHub storage is unavailable, the pinned public dataset can be recovered
directly before running the data checks:

```sh
uv run --frozen --group data python -m taed2_safeagent.dataset fetch
uv run --frozen --group data dvc repro validate
```

This checks the source version and file contents before preparing the data.
It does not recover the saved encoder weights. The
[Dataset Card](docs/dataset-card.md) explains the source, cleaning, grouped split,
input formats and known limits. The [setup guide](docs/getting-started.md)
has more details about storage and reproduction.

## Train the first model

The baseline uses character TFIDF and a linear SVM with command inputs.
After installing the full environment, run its stages separately with:

```sh
uv run --frozen --group data --group training dvc repro train
uv run --frozen --group data --group training dvc repro evaluate
```

The [baseline guide](docs/baseline.md) explains model recovery and output files.
The [encoder guide](docs/encoder-training.md) covers CodeBERT and ModernBERT
with both input modes, including GPU training.

The first baseline experiment reached validation macro F1 0.8613 and DENY recall
0.8613, with 19 DENY commands predicted as ALLOW. Its
[original run](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/80c83cb3105a428db83b5038ab37c0ec)
is still available. The [current saved validation results](reports/baseline/metrics.json)
record macro F1 0.8743 and DENY recall 0.9424, with 11 such errors.

The [final research comparison](reports/test_evaluation/test_comparison_summary.json)
also includes evaluation on the separate test partition:

| Model | Test macro F1 | Test DENY recall | DENY predicted as ALLOW |
| --- | --- | --- | --- |
| Calibrated linear SVM | 0.8375 | 0.7814 | 96 |
| ModernBERT champion with context | 0.8922 | 0.9411 | 13 |

The deployment SVM is a separate fit using all 33,325 prepared examples.
It has no independent test score. The research comparison keeps its own saved
metrics and predictions.

### Record a shared experiment

Follow the [experiment setup](docs/baseline.md#record-an-experiment) to configure
MLflow credentials, then run from a clean committed checkout:

```sh
uv run --frozen --group data --group training python -m taed2_safeagent.modeling.experiment --params params.yaml
```

This trains the baseline, evaluates validation, uploads the model through DVC
and records the run in DagsHub. The run link is saved in
reports/baseline/mlflow_run.json. Normal DVC reproduction does not create a
shared MLflow run. Training can still record local CodeCarbon measurements.

## Energy consumption and emissions

[CodeCarbon](https://codecarbon.io/) records training and inference energy.
The tables keep the results available in the repository. Emissions are shown
in grams of CO2 equivalent for training and milligrams for each prediction.

### Training energy by model

| Model | Input mode | Hardware | Recorded duration | Energy (kWh) | Emissions (g CO2eq) | MLflow |
| --- | --- | --- | --- | --- | --- | --- |
| TFIDF + Linear SVM | command | CPU | 8.09 s | 0.0000462 | 0.0171 | [Run](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/a4792de573684040afff11755b24f7fe) |
| CodeBERT | command | Tesla T4 | 478.78 s | 0.0156 | 5.780 | [Run](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/a81f878bbbff4019b032b389a0909c01) |
| CodeBERT | command with context | Tesla T4 | 834.08 s | 0.0273 | 10.073 | [Run](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/c3532eb8a9c34146926bc9d2e10d047a) |
| ModernBERT | command | Tesla T4 | 791.29 s | 0.0259 | 9.555 | [Run](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/4851c0df192344fa99eacf1721bdbafe) |
| ModernBERT | command with context | Tesla T4 | 1353.73 s | 0.0443 | 16.380 | [Run](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/5ed5c367a6c8481d93a744ce97e7a7b0) |

The durations come from the CodeCarbon measurement windows. The baseline
measurement is saved in [its energy report](reports/baseline/energy.json),
and each encoder has its own energy report in reports/encoders.

The Kaggle sessions reported two Tesla T4 cards, while CodeCarbon was configured
to monitor GPU 0. CPU and RAM values include estimates. The USA carbon factor
was assumed because the physical cloud region was not confirmed.

### CPU inference energy

The [inference report](reports/inference/energy.json) uses one CPU thread.
Some entries are measured and others estimate energy from earlier latency
measurements, as shown in the last column.

| Model | Input mode | Median latency | Energy per query | Emissions per query | Evidence |
| --- | --- | --- | --- | --- | --- |
| TFIDF + Linear SVM | command | 0.45 ms | 0.000004 Wh | 0.0016 mg | Measured over 500 queries |
| CodeBERT | command | 831.00 ms | 0.0148 Wh | 5.642 mg | Estimate from earlier latency |
| CodeBERT | command with context | 12,639.40 ms | 0.2249 Wh | 85.458 mg | Estimate from earlier latency |
| ModernBERT | command | 5,390.19 ms | 0.0964 Wh | 36.617 mg | Estimate from earlier latency |
| ModernBERT | command with context | 616.88 ms | 0.0070 Wh | 2.597 mg | Measured over 3 queries |

The SVM is much cheaper to run on CPU in these measurements, which supports
its use for the Virtech demo. These are model measurements on controlled
batches. API requests also include HTTP and validation work.

## Project layout

| Folder or file | Contents |
| --- | --- |
| taed2_safeagent/data | Source checks, audit, groups, splits, validation and input builder |
| taed2_safeagent/api | Request schemas, model loading and HTTP routes |
| taed2_safeagent/dataset.py | Data commands |
| taed2_safeagent/features.py | Character features for the baseline |
| taed2_safeagent/modeling | Training, evaluation, tracking and diagnostics |
| data/raw/shell_safety | Original source snapshot stored with DVC |
| data/interim/shell_safety | Clean inputs, provenance and quarantine stored with DVC |
| data/processed/shell_safety | Prepared splits and manifest stored with DVC |
| reports/data | Audit, split and validation reports |
| models/tfidf_char_linear_svm | Research baseline stored with DVC |
| models/vm_linear_svm | SVM used by the API, stored with DVC |
| models/encoders | Saved CodeBERT and ModernBERT models stored with DVC |
| reports/baseline | Baseline metrics, predictions and energy |
| reports/encoders | Encoder metrics, energy and experiment links |
| reports/test_evaluation | Final research test comparison |
| scripts | VM setup, service commands and HTTP checks |
| notebooks | Local and Kaggle baseline notebooks |
| tests | Data, model and API tests |
| docs | Setup guides, Dataset Card, model cards and team workflow |
| params.yaml | Data protocol and model settings |
| dvc.yaml and dvc.lock | Pipeline stages and recorded artifact versions |
| pyproject.toml and uv.lock | Dependencies and locked environment |

The project started from Cookiecutter Data Science.

## Team workflow

The [team workflow guide](docs/team-workflow.md) links Trello tasks to branches
and pull requests, includes a dated board export and explains Slack coordination
and the GitHub notification setup. The [feature ownership guide](docs/feature-ownership.md)
records maintenance and review roles.

## Quality checks

For the quick API checks, which need no downloaded model:

```sh
uv run --frozen --group dev pytest tests/test_api.py tests/test_api_model.py tests/test_api_demo.py -q
```

For the full project, install the training group and prepare the data first:

```sh
uv run --frozen --group data --group dev --group training pytest -q
uv run --frozen --group data --group dev --group training pylint taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling taed2_safeagent/api tests kaggle_train.py
uv run --frozen --group data --group dev mkdocs build --strict
```

GitHub Actions runs the project checks in one job, using the public pinned
dataset without personal DagsHub credentials. Some tests need saved data or
encoder files and are reported as skipped when those files are missing.
Tracking tests use a fake client. PyNBLint is available for notebook checks.

Use make help for the optional convenience commands. The uv commands also
work without make.
