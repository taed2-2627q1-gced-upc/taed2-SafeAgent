# Encoder training

CodeBERT and ModernBERT use the same prepared data as the baseline. Each model
has a command run and a command with context run, so each pair changes only the
input mode. Test stays reserved and validation chooses the best saved epoch.

The backbone revisions and training settings are in params.yaml. Initial runs
use two epochs, learning rate 0.00002, effective batch 32 and a 512 token limit.
Padding follows each batch and longer inputs are truncated from the right.
The command comes before the safe context fields, and saved counts show how
often truncation occurs. Labels and annotation fields never become inputs.

## Run a model

Install the training group and prepare the data first:

```sh
uv sync --frozen --group data --group dev --group training
uv run --frozen --group data dvc repro validate
uv run --frozen --group data --group training python -m taed2_safeagent.modeling.encoder codebert /path/to/new/results --mode command
```

Use modernbert for the other backbone and command-context for its paired input
mode. The output directory must be new. A normal run requires clean committed
source and a GPU, while --allow-cpu is available for small development checks.

The trainer uses the pinned
[CodeBERT checkpoint](https://huggingface.co/microsoft/codebert-base) and
[ModernBERT checkpoint](https://huggingface.co/answerdotai/ModernBERT-base).
It disables external tracking callbacks, so fitting does not create a local or
shared MLflow run by itself.

## Train on Kaggle

The repository includes kaggle_train.py for a private GPU script session.
Copy that script to a separate upload folder and set GIT_COMMIT to its published
source version. Set the real Kaggle reference and next saved version as well,
then confirm the version using the API after submission.

The script installs the frozen Python 3.11 environment, fetches the pinned public
dataset and runs the data quality stages. It needs no embedded Git, DVC or MLflow
credentials. It uses one visible GPU and runs both modes of both backbones,
saving each result separately. A failed fit is recorded and the remaining runs
continue. The fitted files and session record stay in saved Kaggle outputs.

Use private metadata with Internet and GPU enabled, selecting the available
accelerator through the official CLI. The checked account supports Tesla T4.
Only the script and kernel metadata belong in the upload folder.

## Saved results

Each output contains the fitted model and tokenizer, source and data identity,
settings, input DVC lock, truncation counts, validation metrics and aligned
predictions. Model selection uses macro F1 and the report also shows class
results, the confusion matrix and DENY examples predicted as ALLOW.

CodeCarbon covers fitting and epoch validation. Its report keeps duration,
energy in kWh, emissions in kg CO2e and the available hardware details. CPU and
RAM readings are treated as estimates unless sensor support is verified. The
USA carbon factor is an explicit assumption for the cloud run and its physical
region remains unverified. An unavailable monitor is reported without a made up
zero, and fitting failures still stop the monitor.

Cloud results need their actual Kaggle session and saved version recorded.
After download, their data and model versions must be checked before DVC upload
and shared MLflow registration. Registration must retain the original execution
source and timestamps, separately from the commit containing returned results.

Use the registration command after confirming the saved session version:

```sh
uv run --frozen --group data python -m taed2_safeagent.modeling.register /path/to/model/result /path/to/session.json
```

Registration checks the saved validation predictions and configuration, uploads
the model through DVC and records the run in shared MLflow. Small metrics and
run links go in reports/encoders. Commit each completed result before registering
the next one, keeping its original training source and separate upload source.
