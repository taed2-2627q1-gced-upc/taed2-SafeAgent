# Getting started

Install Git and uv, then clone the GitHub repository and open its root folder.
The locked environment uses Python 3.11. uv can install that interpreter.

```sh
uv sync --frozen --group data --group dev
```

The data group includes Great Expectations and Deepchecks. Keep the frozen
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
Keep credentials out of Git and command history. Do not reinitialize DVC.

```sh
uv run --frozen --group data dvc pull
uv run --frozen --group data dvc repro
uv run --frozen --group data pytest -q
```

A normal reproduction reuses the raw snapshot and valid stage outputs.
Use dvc repro --force when checking that transformations recreate their outputs.
Small reports belong in Git. Large artifacts use DVC.

## Recover from the public source

If DagsHub access is unavailable, fetch the pinned public files instead:

```sh
uv run --frozen --group data python -m taed2_safeagent.dataset fetch
uv run --frozen --group data dvc repro --force
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
uv run --frozen --group data ruff check taed2_safeagent/data taed2_safeagent/dataset.py tests
uv run --frozen --group data ruff format --check taed2_safeagent/data taed2_safeagent/dataset.py tests
uv run --frozen --group data pylint taed2_safeagent/data taed2_safeagent/dataset.py tests
uv run --frozen --group data pytest -q
uv run --frozen --group data mkdocs build --strict
```

The full dataset integration test needs prepared data. Without it, that check is
reported as skipped. CI prepares the actual snapshot before running all tests.
PyNBLint becomes applicable when notebooks are added. This data pipeline does
not train a model or create MLflow and CodeCarbon records.
