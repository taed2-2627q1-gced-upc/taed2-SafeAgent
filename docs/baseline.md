# First baseline

Character TF-IDF and a linear SVM form the first comparison point. This
model reads the command only, so context can be added in a later experiment
using the same examples and split. It never executes commands.

## Run training and evaluation

Follow [getting started](getting-started.md) to install the locked environment
and configure DVC, then run from the repository root:

```sh
uv run --frozen --group data dvc pull
uv run --frozen --group data dvc repro
```

Training fits the vocabulary and classifier on the 26,660 training examples.
Evaluation loads that fitted pipeline and predicts the 3,333 validation examples.
The 3,332 test examples stay reserved for the final comparison. There is no
parameter search in this run.

DVC runs audit, prepare and validate before train and evaluate. The model stage
depends on the quality report, prepared training data, source code, locked
environment and baseline settings. Evaluation depends on the saved model and
validation data. A failed stage stops the following stages.

Current stages are skipped, so a normal reproduction can recover or reuse the
saved model. Metadata keeps the code and fit details from the run that produced
that version. To force a new fit without rerunning the data stages:

```sh
uv run --frozen --group data dvc repro --force --single-item train
uv run --frozen --group data dvc repro evaluate
```

For a quick development run, the Python train and evaluate commands remain
available. A new fit clears the old shared run receipt. After running both
commands directly, record their outputs with dvc commit --force train evaluate.
Normal pipeline runs record outputs automatically.

The baseline settings are in params.yaml. Character lengths are 2 to 5 and
features preserve case, repeated spaces and punctuation. The vocabulary has a
limit of 100,000 features, with min_df 2, sublinear TF, L2 normalization and
float32 values. The SVM uses C 1, balanced classes, seed 42, dual auto, tolerance
0.0001 and at most 10,000 iterations. Training fails if the SVM does not converge.

Only the shared command input builder supplies features. Context, labels,
identifiers and annotation fields do not enter the vectorizer. Evaluation also
checks the saved data identity and rejects overlapping IDs or groups.

## Record an experiment

The DVC stages and separate Python commands work locally. To record a real
shared run, set MLFLOW_TRACKING_USERNAME to the personal DagsHub username and
MLFLOW_TRACKING_PASSWORD to the personal access token. Use the local environment
or an ignored .env file, and keep the values out of Git and command history.
The URI and the shared experiment name SafeAgent are in params.yaml.
An existing MLFLOW_TRACKING_URI must match that URI.

Commit the source and settings, check that DVC inputs are current, then run:

```sh
uv run --frozen --group data python -m taed2_safeagent.modeling.experiment --params params.yaml
```

The command requires a clean Git checkout. It trains the fixed baseline and
evaluates validation, then saves the run identity in model metadata and uploads
the new model version with DVC. MLflow receives settings, metrics, class results,
raw and normalized confusion matrices, prepared ID predictions and the input and
output DVC locks. The model binary stays in DVC, owned by the train stage.
The command records the stages after actually running training and evaluation,
then uploads the new model. Cached reproduction creates no shared run.

After successful logging, the run link and version details are saved in
reports/baseline/mlflow_run.json. Review and commit the updated lock and receipt
afterward. The receipt records the code used during training, separately from
the later commit containing generated results.

A failed run returns an error and does not write a new success receipt. Its
shared status is marked failed when the server is reachable. Generated files
remain available for checking before another attempt. There is no local tracking
fallback and the command does not retry training automatically.

## Saved files

| Path | Contents |
| --- | --- |
| models/tfidf_char_linear_svm/pipeline.joblib | Fitted vectorizer and classifier together |
| models/tfidf_char_linear_svm/metadata.json | Settings, data version, code version and fit details |
| reports/baseline/metrics.json | Validation metrics and confusion matrix |
| reports/baseline/predictions.jsonl | Prepared ID, true label and predicted label for each example |
| reports/baseline/mlflow_run.json | Successful shared run link and code and model versions |

The model directory is stored through the train entry in dvc.lock. The small reports are
stored in Git. dvc pull recovers the saved model, so training can be skipped when
you only want to evaluate it. Load serialized models only from a trusted source.

After a pipeline run, upload the model before committing the updated lock and
reports. The experiment command already performs the upload:

```sh
uv run --frozen --group data dvc push train
```

The old standalone model pointer is replaced by the train stage, so the model
has one owner in DVC. Earlier committed versions remain available in history.

## First result

The [recorded validation run](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/1b2eb37037e34d82ad87c5ba86613c99)
has macro F1 0.8613, accuracy 0.8623 and DENY recall
0.8613. It predicts ALLOW for 19 of the 764 DENY examples, which is 2.49%.
These results give the project a starting point, but the data is synthetic and the model
does not use context. See the [model card](model-cards/model_card_tfidf_linear_svm.md)
for all class results and limits.

This is a local CPU baseline. Shared tracking uses the experiment command above,
and energy measurements remain planned for Milestone 3. No energy or emissions
value has been measured for this run.
