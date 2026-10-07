---
model_id: taed2-safeagent/tfidf-char-linear-svm
language:
  - en
library_name:
  - scikit-learn
pipeline_tag: text-classification
tags:
  - shell-command-safety
  - command-risk-classification
  - multiclass-classification
  - classical-baseline
dataset:
  - tomngdev/shell-safety-v1.1
status: validation baseline
license: other
license_name: Project artifact license to be defined
---

# Model Card for TF-IDF Character n-grams + Linear SVM

This is the first trained SafeAgent baseline. It predicts `ALLOW`, `ASK` or
`DENY` from a proposed shell command using character TF-IDF and a linear SVM.
The results below come from validation only. The final test is still reserved.

## Model Details

### Model Description

The small classical model is a starting point for the later encoder
experiments. It reads the command string only, so it can also help measure
whether adding execution context improves the same model family.

- **Developed by:** SafeAgent project team, TAED2 course project
- **Funded by:** Academic course project
- **Shared by:** SafeAgent project team
- **Model type:** Character TF-IDF followed by LinearSVC
- **Language(s):** Shell commands and command related text
- **License:** Project artifact license still to be defined. The source dataset is MIT licensed and its terms apply separately.
- **Finetuned from model:** None, trained from scratch

### Model Sources

- **Repository:** [taed2-SafeAgent on GitHub](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent)
- **Dataset:** [tomngdev/shell-safety-v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1)
- **Implementation:** [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- **Demo:** Not available yet

## Uses

### Direct Use

The model returns one project label for each command:

- `ALLOW`: the dataset labels the command as suitable to proceed.
- `ASK`: the dataset labels the command as needing confirmation.
- `DENY`: the dataset labels the command as unsuitable to run automatically.

It classifies text and never executes commands. A prediction does not prove
that an action is safe.

### Downstream Use

Use it as a comparison point for later models and for reviewing classification
errors. A future service still needs its own policy checks and confirmation flow.

### Out-of-Scope Use

The baseline is not ready to act as the only safety control in a real agent.
It is not a shell interpreter, sandbox or authorization system.

## Bias, Risks, and Limitations

- The data is synthetic, so repeated patterns and label rules may not match real traffic.
- Character features can learn command fragments that do not generalize to new tools or syntax.
- The same command can have different risk depending on context, which this model does not read.
- A false ALLOW can miss a harmful action, while a false ASK or DENY can interrupt useful work.
- Grouping reduces duplicate leakage but does not prove that all synthetic templates are independent.
- The predictions are class labels. SVM scores are not calibrated probabilities.
- These are validation results from one fixed configuration, with no final test evaluation yet.

### Recommendations

Review DENY recall together with DENY examples predicted as ALLOW. Later work
should inspect errors and compare command and context inputs under the same
model family and split. Do not tune on the final test partition.

## How to Get Started with the Model

Follow the [baseline guide](../baseline.md) to recover the fitted model with DVC
or train it locally. Load model files only from a trusted source.

```python
from joblib import load

pipeline = load("models/tfidf_char_linear_svm/pipeline.joblib")
prediction = pipeline.predict(["git status"])[0]
print(prediction)
```

The fitted vectorizer and classifier are stored together. This example predicts
a label and does not run the command.

## Training Details

### Training Data

The [Dataset Card](../dataset-card.md) records the source and preparation.
The pinned revision is fee89770c315d525ef2ee42adee6ef9725a7621e of
tomngdev/shell-safety-v1.1. Its 34,007 source rows become 33,325 prepared rows
after removing redundant copies and quarantining two conflicting examples.
The strict group split has 26,660 train, 3,333 validation and 3,332 test rows.

Only command text enters the features through the shared input builder.
Context stays available for future comparisons. Reason, category, assistant
text, shell tags, IDs and source metadata are excluded from predictive inputs.

### Training Procedure

#### Preprocessing

The custom character analyzer preserves case, repeated spaces and punctuation.
It creates character fragments of lengths 2 to 5. Vocabulary and inverse
document frequencies are fitted on training text only. There is no stemming
or stop word removal.

#### Training Hyperparameters

| Component | Setting |
| --- | --- |
| TF-IDF | min_df 2, max_features 100000, sublinear_tf true, norm l2, float32 |
| LinearSVC | C 1, class_weight balanced, random_state 42, dual auto |
| Solver limits | tol 0.0001, max_iter 10000 |

The fixed configuration is in params.yaml. No parameter search was run.
The classifier converged in 61 iterations and the vocabulary has 100,000 features.

#### Speeds, Sizes, Times

Fitting the vectorizer and classifier took 4.30 seconds on a local Intel Core
i7-1255U CPU. This timing excludes data loading and saving the model.
The compressed pipeline is 2,124,861 bytes, about 2.03 MiB.
Inference latency and peak memory have not been benchmarked.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

This first evaluation uses the 3,333 validation examples only. The test
partition is reserved for the final comparison. Evaluation loads the saved
pipeline and checks that the data identity matches and that IDs and groups
do not overlap with training.

#### Factors

Results are reported by target class. Further analysis by command length,
context availability and source category remains future work. Those fields
must stay outside the feature input for this command baseline.

#### Metrics

The reports include accuracy, balanced accuracy, macro F1, per class precision, recall,
F1 and support, and the confusion matrix. They also count true DENY examples
predicted as ALLOW and divide that count by all true DENY examples.

### Results

Recorded validation run on 2026-10-07, available in
[MLflow on DagsHub](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/1b2eb37037e34d82ad87c5ba86613c99):

| Metric | Value |
| --- | --- |
| Accuracy | 0.8623 |
| Balanced accuracy | 0.8595 |
| Macro F1 | 0.8613 |
| DENY recall | 0.8613 |
| DENY predicted ALLOW | 19 of 764, or 2.49% |

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| ALLOW | 0.8672 | 0.9265 | 0.8958 | 1360 |
| ASK | 0.8415 | 0.7907 | 0.8154 | 1209 |
| DENY | 0.8844 | 0.8613 | 0.8727 | 764 |

Confusion matrix, with true classes in rows and predictions in columns:

| True / Predicted | ALLOW | ASK | DENY |
| --- | --- | --- | --- |
| ALLOW | 1260 | 93 | 7 |
| ASK | 174 | 956 | 79 |
| DENY | 19 | 87 | 658 |

The full metrics are in reports/baseline/metrics.json. The predictions file
uses prepared IDs to support later error analysis. Model settings, data
identity and the clean source commit are recorded in the saved metadata.

#### Summary

This gives the project a useful first comparison point, but the 19 DENY to ALLOW errors
still matter. The results do not establish safety on real agent traffic.

## Model Examination

ASK has the lowest class recall in this run. Error review and examination of
linear feature weights have not been completed yet. Weights would describe
patterns learned from this dataset, rather than explanations of command safety.

## Environmental Impact

Energy and carbon emissions were not measured in this run. CodeCarbon work
is planned for Milestone 3. No zero emissions value is assumed.

- **Hardware Type:** Local CPU
- **Hours used:** Energy tracking not performed
- **Cloud Provider:** No cloud used for this run
- **Carbon Emitted:** Not measured

## Technical Specifications

### Model Architecture and Objective

```text
command text
    -> raw character TF-IDF features
    -> linear SVM classifier
    -> ALLOW / ASK / DENY
```

The sparse TF-IDF matrix feeds a multiclass LinearSVC with balanced class
weights. The model does not provide predict_proba or an uncertainty routing policy.

### Compute Infrastructure

#### Hardware

The first run used a local Intel Core i7-1255U CPU. No GPU was used.
Peak RAM use has not been measured.

#### Software

Python 3.11, scikit-learn 1.5.2, NumPy 1.26.4 and joblib 1.6.0, with the
environment fixed in uv.lock. DVC stores the fitted model. Shared MLflow tracking
uses the experiment command in the [baseline guide](../baseline.md). Its receipt
records the completed run link. Energy tracking remains planned for Milestone 3.

## Citation

There is no project paper for this baseline. Refer to the dataset and the
implementation links above when reusing it.

## Glossary

- **TF-IDF:** Character frequencies weighted by how common they are across training examples.
- **Linear SVM:** A classifier that separates classes in the feature space.
- **Leakage:** Information that makes evaluation easier without being available at prediction time.

## More Information

See the [baseline guide](../baseline.md) for commands and saved file paths.

## Model Card Authors

SafeAgent project team, TAED2 course project.

## Model Card Contact

Use the [repository issue tracker](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/issues).
