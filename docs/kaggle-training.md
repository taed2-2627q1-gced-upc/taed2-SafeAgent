# Training on Kaggle

Kaggle is the selected environment for later CodeBERT and ModernBERT training.
The existing character TF-IDF and Linear SVM baseline uses CPU, so its setup
does not need GPU quota. GitHub stores source, DagsHub stores DVC assets, and
the shared DagsHub MLflow experiment records formal runs.

The [baseline guide](baseline.md) explains the implemented model and its local
validation result. Transformer training remains planned. No SafeAgent training
or GPU allocation on Kaggle has been verified.

## Local access

Install the official CLI outside the project's environment and sign in through
the browser:

```powershell
uv tool install --python 3.11 kaggle==2.2.4
kaggle --version
kaggle auth login
kaggle kernels list --mine --page-size 5
kaggle quota --format json
```

If the command is unavailable, `uv tool dir --bin` locates its directory.
The [official authentication guide](https://github.com/Kaggle/kaggle-cli/blob/v2.2.4/docs/README.md#authentication)
also describes token authentication. Credentials belong outside Git, notebooks
and logs.

On 2026-10-07, CLI 2.2.4 and authenticated notebook listing passed. The quota
API reported 30 GPU hours remaining and zero used, with a refresh value of
`2026-10-10T00:00:00`. That response gives no timezone. Quota is a dated
snapshot, so access and availability need checking before a later GPU run.
The checked page contained 19 owned notebooks and no SafeAgent match.
The Kaggle account is `joelmrquezalvarez`, which differs from the GitHub account.

## Notebook setup

The [CPU baseline notebook](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/blob/main/notebooks/kaggle_baseline.ipynb)
contains the setup and the two execution paths. Set its full published source
commit, actual notebook reference and saved source version before running it.
For a new submission, enter its intended next version and confirm it on the
saved run page afterward. A mismatch needs correction in the downloaded session
record before using that result as experiment evidence.
The exported session marks its notebook version as pending confirmation.

Use a private Python notebook with Internet enabled and Accelerator set to None
for the CPU baseline. Internet is needed to clone GitHub, install the frozen
environment and reach DVC and MLflow.

Create and enable the following entries under Add-ons > Secrets. Copy the DVC
values from the project's DagsHub storage setup, using credentials for an account
with access to that repository.

| Kaggle Secret | Purpose |
| --- | --- |
| DVC_ACCESS_KEY_ID | S3 access key for the DagsHub DVC remote |
| DVC_SECRET_ACCESS_KEY | S3 secret key for the DagsHub DVC remote |
| MLFLOW_TRACKING_USERNAME | Personal DagsHub username for shared logging |
| MLFLOW_TRACKING_PASSWORD | Personal DagsHub access token for shared logging |

The baseline setup needs the two DVC secrets. The MLflow secrets are also needed
when recording a new shared experiment. Load them at runtime through
`kaggle_secrets.UserSecretsClient`. Kaggle Secrets must be configured in the
notebook editor, as described in the [official kernel guide](https://github.com/Kaggle/kaggle-cli/blob/v2.2.4/docs/kernels.md#using-secrets-in-kernels).
Never copy a local credential file into the upload folder or save secret values
in notebook outputs.

## Reproducible source and data

Start from a published full Git commit, then clone that version into a writable
directory under `/kaggle/temp`. Reusable code stays in `taed2_safeagent`.
Kaggle's installed packages do not replace the project's Python 3.11 environment
or `uv.lock`. From that checkout, install and recover the prepared data:

```sh
uv sync --frozen --python 3.11 --group data
uv run --frozen --group data dvc pull
uv run --frozen --group data dvc repro validate
```

The final command checks the data stages without starting model training.
The prepared split has 26,660 train, 3,333 validation and 3,332 test examples.
Training fits only on train and evaluation uses validation. Test stays reserved
for the final comparison. The [dataset card](dataset-card.md) describes the
pinned source, duplicate handling and group split.

The baseline reads the command through the shared input builder. Labels,
annotation reason and category never become predictive features. Context is a
separate future comparison using the same prepared examples.

## Baseline and shared tracking

The development baseline uses the existing package commands:

```sh
uv run --frozen --group data python -m taed2_safeagent.modeling.train --params params.yaml
uv run --frozen --group data python -m taed2_safeagent.modeling.evaluate --params params.yaml
```

These commands train and evaluate without a new shared MLflow run. A formal run
starts from a clean committed checkout and uses the explicit experiment command
instead:

```sh
uv run --frozen --group data python -m taed2_safeagent.modeling.experiment --params params.yaml
```

The shared experiment is `SafeAgent` at
`https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow`, as configured in
`params.yaml`. The command requires the MLflow credentials and uploads the model
through DVC. A failed remote operation is an error, with no local tracking fallback.
The run receipt is `reports/baseline/mlflow_run.json`. Recovered receipts belong
to their original runs and are not evidence of a new Kaggle experiment.

The notebook exports selected files to `/kaggle/working/safeagent-results`.
The model, reports, settings and `session.json` identify the result, with the
recovered input version in `input_dvc.lock`. A formal run also exports its new
`output_dvc.lock` and the model version metadata named by its run receipt in the
original relative path. Development training does not version or upload its new model.
The temporary checkout, installed environment and caches are not exported.

## Save source and results

Prepare an upload directory outside the checkout containing only the notebook
and its metadata file. Initialize that file with:

```powershell
kaggle kernels init -p <upload-folder>
```

Set the actual Kaggle account and notebook slug in `id`, the notebook filename
in `code_file`, `language` to `python`, and `kernel_type` to `notebook`. For the
baseline, set `is_private` and `enable_internet` to true and `enable_gpu` to false.
Attached data, competition, kernel and model sources can stay empty because DVC
recovers the inputs. See the [metadata specification](https://github.com/Kaggle/kaggle-cli/blob/v2.2.4/docs/kernels_metadata.md).

The following commands are templates. Replace their angle bracket values first.
Pushing a notebook starts remote execution, so the upload is a separate step
after its source and secrets are ready.

```powershell
kaggle kernels push -p <upload-folder>
kaggle kernels status <account>/<notebook-slug>
kaggle kernels output <account>/<notebook-slug> -p <local-output-folder>
```

Record the full execution Git commit, input DVC version, notebook reference and
actual saved source version with the results. A formal run also needs the shared
MLflow run link and later output commit recorded separately. Review generated
models, metrics and run receipts before publishing output metadata, and confirm
the required DVC upload. Kaggle outputs are a way to transfer results.

## Local notebook checks

Local checks passed for notebook format, Python syntax, both execution paths,
secret redaction, command failures and selected result export. Ruff passed and
Pylint checked the extracted cells with an allowance for the Kaggle secrets
import, which is only available on the notebook host.

PyNBLint 0.1.6 ran with findings. The notebook is unexecuted, and the scan of the
notebook folder reports missing Git, dependency and coverage metadata. Git and
`pyproject.toml` are at the project root, and coverage was not collected.
Cloud execution remains pending.

## Later GPU training

CodeBERT and ModernBERT need separate implemented training code and locked
dependencies before a GPU notebook is ready. Select an accelerator available to
the account and check CUDA, actual GPU names and memory inside that session.
Validate batch size, sequence length and precision before a formal benchmark.
Multiple visible GPUs need explicit support in the training code.

Stop interactive sessions after use and save checkpoints before session limits.
Record hardware and software versions with timing and energy measurement limits.
No SafeAgent energy or emissions result is available from this setup.
