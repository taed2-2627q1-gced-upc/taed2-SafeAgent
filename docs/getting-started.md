# Getting started

Install Git and uv, then clone the GitHub repository and open its root folder.
The locked environment uses Python 3.11. uv can install that interpreter.

```sh
uv sync --frozen --group data --group dev
```

This environment runs the API, data checks and CPU baseline with CodeCarbon.
For encoder training, install the neural model libraries as well:

```sh
uv sync --frozen --group data --group dev --group training
```

The data group includes Great Expectations, Deepchecks and CodeCarbon. Keep the frozen
lockfile rather than updating libraries during reproduction. PyNBLint requires
Typer 0.12.5 and Click 8.1.8. Deepchecks requires the pinned scientific stack and
setuptools 80.9.0. These limits avoid the import and CLI failures found during
setup. The libraries still emit upstream deprecation warnings.

## Windows cache

A long OneDrive checkout path can exceed the Windows path limit in the DVC run
cache. Set a short local cache path before using DVC. This changes local config
only and does not change the shared remote or credentials.

```powershell
uv run --frozen --group data dvc config --local cache.dir "$env:LOCALAPPDATA/SafeAgent/dvc-cache"
```

## Recover versioned data

The shared remote is storage, using s3://dvc and this DagsHub endpoint:
https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.s3.
Use the personal S3 credentials provided by the DagsHub storage page. Configure
access_key_id and secret_access_key with dvc remote modify --local, or use
AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.
Keep credentials out of Git. The remote is already configured, so DVC does not
need to be initialized again.

After replacing the placeholders with the personal storage credentials:

```sh
uv run --frozen dvc remote modify --local storage access_key_id YOUR_ACCESS_KEY
uv run --frozen dvc remote modify --local storage secret_access_key YOUR_SECRET_KEY
```

The values are saved in .dvc/config.local, which Git ignores.

## Run the saved API model

Only the deployment SVM is needed for the API:

```sh
uv run --frozen dvc pull models/vm_linear_svm
uv run --frozen uvicorn taed2_safeagent.api.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs and keep the terminal open while using it.
The [API guide](api.md) explains the request fields and model configuration.

## Recover all data and models

For the full project, install the training group before reproducing stages:

```sh
uv run --frozen --group data dvc pull
uv run --frozen --group data --group training dvc repro
uv run --frozen --group data --group training pytest -q
```

A normal reproduction reuses the raw snapshot and valid stage outputs.
Use dvc repro --force when checking that transformations recreate their outputs.
Small reports belong in Git. Large artifacts use DVC.
The pipeline trains the research baseline and evaluates validation. A separate
train_vm stage fits the deployment SVM on all prepared partitions. Its artifact
has no independent test score. A full pull also includes the saved encoder models.
Use dvc repro validate to prepare and check only the data.

## Recover from the public source

If DagsHub access is unavailable, fetch the pinned public files instead:

```sh
uv run --frozen --group data python -m taed2_safeagent.dataset fetch
uv run --frozen --group data dvc repro validate
```

Fetch checks the source revision, file sizes, SHA256 hashes and row counts.
It reuses a verified snapshot and refuses to overwrite a corrupt or different
snapshot. Investigate a mismatch rather than deleting or editing raw data.
A source update needs an explicit new revision, hashes and raw snapshot path.

The CLI also supports audit, prepare and validate, with --params for the parameter
file. Normal runs should use DVC so dependencies and output versions are recorded.
A failed validation exits with an error and removes a stale success report.

## Checks and docs

```sh
uv run --frozen --group data --group dev ruff check taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling taed2_safeagent/api tests kaggle_train.py
uv run --frozen --group data --group dev ruff format --check taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling taed2_safeagent/api tests kaggle_train.py
uv run --frozen --group data --group dev --group training pylint taed2_safeagent/data taed2_safeagent/dataset.py taed2_safeagent/features.py taed2_safeagent/modeling taed2_safeagent/api tests kaggle_train.py
uv run --frozen --group data --group dev --group training pytest -q
uv run --frozen --group data --group dev mkdocs build --strict
```

The full dataset integration test needs prepared data. Without it, that check is
reported as skipped. CI prepares the actual snapshot before running all tests.
PyNBLint is available for checking the included notebooks. Follow the
[baseline guide](baseline.md) for local training and validation evaluation.
Normal reproduction does not create a shared MLflow run. Training can record
local CodeCarbon measurements in the configured report folder.
