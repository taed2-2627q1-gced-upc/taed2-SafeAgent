# First baseline

We use character TF-IDF and a linear SVM as the first comparison point. This
model reads the command only, so context can be added in a later experiment
using the same examples and split. It never executes commands.

## Run training and evaluation

Follow [getting started](getting-started.md) to install the locked environment
and configure DVC, then run from the repository root:

```sh
uv run --frozen --group data dvc pull
uv run --frozen --group data dvc repro
uv run --frozen --group data python -m taed2_safeagent.modeling.train --params params.yaml
uv run --frozen --group data python -m taed2_safeagent.modeling.evaluate --params params.yaml
```

Training fits the vocabulary and classifier on the 26,660 training examples.
Evaluation loads that fitted pipeline and predicts the 3,333 validation examples.
The 3,332 test examples stay reserved for the final comparison. There is no
parameter search in this run.

The baseline settings are in params.yaml. Character lengths are 2 to 5 and
features preserve case, repeated spaces and punctuation. The vocabulary has a
limit of 100,000 features, with min_df 2, sublinear TF, L2 normalization and
float32 values. The SVM uses C 1, balanced classes, seed 42, dual auto, tolerance
0.0001 and at most 10,000 iterations. Training fails if the SVM does not converge.

Only the shared command input builder supplies features. Context, labels,
identifiers and annotation fields do not enter the vectorizer. Evaluation also
checks the saved data identity and rejects overlapping IDs or groups.

## Saved files

| Path | Contents |
| --- | --- |
| models/tfidf_char_linear_svm/pipeline.joblib | Fitted vectorizer and classifier together |
| models/tfidf_char_linear_svm/metadata.json | Settings, data version, code version and fit details |
| reports/baseline/metrics.json | Validation metrics and confusion matrix |
| reports/baseline/predictions.jsonl | Prepared ID, true label and predicted label for each example |

The model directory is stored through its DVC pointer. The small reports are
stored in Git. dvc pull recovers the saved model, so training can be skipped when
you only want to evaluate it. Load serialized models only from a trusted source.

After a new training run, version and upload the model before committing its
updated pointer and reports:

```sh
uv run --frozen --group data dvc add models/tfidf_char_linear_svm
uv run --frozen --group data dvc push models/tfidf_char_linear_svm.dvc
```

Training and evaluation currently use the two Python commands above. They are
not stages in dvc.yaml yet, so dvc repro runs the data pipeline only.

## First result

The first validation run has macro F1 0.8613, accuracy 0.8623 and DENY recall
0.8613. It predicts ALLOW for 19 of the 764 DENY examples, which is 2.49%.
These results give us a starting point, but the data is synthetic and the model
does not use context. See the [model card](model-cards/model_card_tfidf_linear_svm.md)
for all class results and limits.

This is a local CPU baseline. Shared MLflow tracking is planned for the next PR
and energy measurements for Milestone 3. No energy or emissions value has been
measured for this run.
