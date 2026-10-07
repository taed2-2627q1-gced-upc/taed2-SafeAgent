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
  - draft
dataset:
  - tomngdev/shell-safety-v1.1
status: draft
license: other
license_name: Project artifact license to be defined
---

# Model Card for TF-IDF Character n-grams + Linear SVM

This is the initial classical baseline for SafeAgent. It classifies a proposed
shell command into `ALLOW`, `ASK`, or `DENY` using character-level TF-IDF
features and a linear support-vector machine. The card describes the planned
model configuration, no trained checkpoint or evaluation result is currently
included in the repository.

## Model Details

### Model Description

SafeAgent is an ML component for estimating the operational risk of shell
commands proposed by an AI coding agent. This model is deliberately small and
interpretable. It uses the command string only, so it provides a controlled
baseline for measuring how far lexical and syntactic patterns can go before
adding contextual encoders.

- **Developed by:** SafeAgent project team, TAED2 course project
- **Funded by:** Academic course project, no external funding identified
- **Shared by:** SafeAgent project team
- **Model type:** `TfidfVectorizer` with character n-grams followed by a linear SVM classifier
- **Language(s):** English shell commands and command-related text
- **License:** The license for the trained project artifact has not yet been declared. The source dataset is MIT-licensed, third-party library and dataset terms must be respected separately.
- **Finetuned from model:** None, trained from scratch on the project dataset

### Model Sources

- **Repository:** [taed2-SafeAgent on GitHub](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent)
- **Dataset:** [tomngdev/shell-safety-v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1)
- **Implementation references:** [TfidfVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [LinearSVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- **Paper:** Not applicable to this project-specific combination
- **Demo:** Not available

## Uses

### Direct Use

Given one proposed shell command, the model returns one of three project
labels:

- `ALLOW`: the command may be executed without additional confirmation,
- `ASK`: the command may have relevant side effects and should require confirmation,
- `DENY`: the command should not be executed automatically.

The prediction is a risk-classification signal only. The model must never
execute the command itself.

### Downstream Use

The model can be used as a lightweight first-stage filter in a coding-agent
workflow, as a comparison point for transformer models, and as a source of
feature-level error analysis. A downstream application should combine it with
independent policy checks, sandboxing, allowlists/denylists, and a human
confirmation path.

### Out-of-Scope Use

This model is not a shell interpreter, sandbox, malware detector, authorization
system, or proof that a command is safe. It should not be used as the only
control for production infrastructure, destructive operations, credentials,
untrusted input, or commands from shells and environments not represented in
the evaluation data. It is also not intended to infer the user's legal,
organizational, or security policy.

## Bias, Risks, and Limitations

- The dataset is synthetic, so its label rules and lexical patterns may not
  represent real coding-agent traffic.
- Character n-grams can memorize command fragments or dataset artifacts and may
  fail on novel tools, obfuscation, aliases, or unusual syntax.
- The command-only input cannot distinguish two identical commands whose risk
  differs because of the working directory, repository state, privileges,
  operating system, or user intent.
- False `ALLOW` predictions can enable harmful actions, false `DENY` or `ASK`
  predictions can interrupt legitimate workflows.
- Near-duplicate commands must not cross train and test splits. Otherwise,
  measured performance may be artificially high.
- `category` and `reason` are intentionally excluded from the predictive
  input because they can leak target information. They may be used for slicing
  and error analysis only.
- The model has no calibrated safety guarantee. A decision score is not a
  probability unless calibration is measured and documented.

### Recommendations

Use `DENY` recall as a primary safety guardrail and report the trade-off with
`ASK`/`ALLOW` usability. Calibrate or threshold scores on a validation set,
route uncertain cases to `ASK`, and fail closed when the classifier or its
features cannot be loaded. Evaluate separately by shell, command length,
context availability, and risk category. Keep an auditable record of the
dataset snapshot, model checksum, prediction, and final human/system decision.
Red-team the deployed workflow with destructive, obfuscated, chained, and
environment-dependent commands.

## How to Get Started with the Model

No fitted artifact is currently published. The following is an illustrative
loading example for the planned serialized scikit-learn pipeline, the path and
label mapping must be updated when training produces a checkpoint.

```python
from joblib import load

pipeline = load("models/tfidf_char_linear_svm.joblib")
command = "rm -rf ./build"
prediction = pipeline.predict([command])[0]
print(prediction)  # ALLOW, ASK, or DENY
```

The code predicts a label, it does not run the command.

## Training Details

### Training Data

The data protocol is fixed in the [Dataset Card](../dataset-card.md).
Use revision fee89770c315d525ef2ee42adee6ef9725a7621e and the strict group
split with 26,660 train, 3,333 validation and 3,332 test rows. Use the shared
input builder and prepared IDs. Predictive text excludes reason, category,
assistant text, shell tags and source metadata. A context ablation must
keep the model family and selection protocol fixed.

The planned training source is [Shell Safety v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1), a synthetic dataset of 34,007 examples with `command`, `session_context`, `label`, `category`, `shell`, and `reason` fields. The project target is `label`, normalized to the uppercase classes `ALLOW`, `ASK`, and `DENY`.

For this baseline, only `command` is used as a predictive feature. `session_context`
is retained for later comparison experiments, while `category` and `reason` are
excluded from the feature matrix to avoid leakage. The exact dataset revision,
split counts, duplicate policy, and class distribution must be recorded with
the first training run.

### Training Procedure

#### Preprocessing

The planned preprocessing preserves punctuation and short character patterns
because shell operators, flags, paths, separators, and executable names can
carry risk information. The vectorizer configuration (n-gram range, minimum
document frequency, normalization, and vocabulary size) must be fixed in the
experiment configuration and stored with the fitted pipeline. No stemming,
natural-language stop-word removal, or use of `category`/`reason` is planned.

#### Training Hyperparameters

- **Training regime:** Sparse CPU training, normally FP32, exact vectorizer and SVM hyperparameters pending the first reproducible run

#### Speeds, Sizes, Times

Not measured yet. This baseline is expected to require substantially less
memory and inference time than the transformer alternatives, but latency,
serialized size, vocabulary size, and training duration must be measured on the
same evaluation hardware before making deployment claims.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Evaluation will use a held-out split from the same versioned dataset, with
duplicate and near-duplicate leakage checks. The final card must record the
dataset revision, split construction, sample counts, and class distribution.

#### Factors

Report results overall and by:

- target class (`ALLOW`, `ASK`, `DENY`),
- shell family, such as POSIX, PowerShell, and CMD when available,
- command-length bins and presence of shell operators,
- whether execution context is present, even though it is not used by this model,
- `category` for analysis-only slices, never as a model input.

#### Metrics

The primary safety metric is recall for `DENY`. Also report macro-F1,
per-class precision/recall/F1, balanced accuracy, overall accuracy, the
confusion matrix, and inference latency. If scores are used for routing to
`ASK`, report calibration metrics and the coverage/risk trade-off.

### Results

No training or evaluation run has been committed yet. Results are therefore
**pending** and must not be inferred from the performance of scikit-learn or
from the source dataset page.

#### Summary

This model is the low-cost reference point for the CodeBERT and ModernBERT
experiments. The project should select a final model using safety metrics,
latency, size, cost, energy, and operational practicality rather than model
complexity alone.

## Model Examination

The fitted linear classifier can support a useful first examination through
feature weights, influential character n-grams, and manually reviewed false
positives/false negatives. Such weights indicate associations in this dataset,
not causal explanations or a guarantee that a command is safe.

## Environmental Impact

Carbon emissions have not been measured for this untrained model. The first
training run should record energy with CodeCarbon where practical.

- **Hardware Type:** Not measured, CPU-oriented baseline expected
- **Hours used:** Not measured
- **Cloud Provider:** Not applicable/not recorded
- **Compute Region:** Not recorded
- **Carbon Emitted:** Not measured

## Technical Specifications

### Model Architecture and Objective

The planned pipeline is:

```text
command string
    -> character-level TF-IDF sparse matrix
    -> linear SVM multiclass classifier
    -> ALLOW / ASK / DENY
```

The vocabulary and number of effective features are learned from the training
split and are not known until the model is fitted. The classifier is intended
to optimize a margin-based multiclass decision function, exact class weighting
and regularization must be documented with the checkpoint.

### Compute Infrastructure

Training and inference should be possible on a standard CPU environment. A GPU
is not required. Reproducibility requires a pinned Python environment, random
seed where applicable, dataset revision, and serialized vectorizer/classifier.

#### Hardware

CPU and RAM requirements are not yet benchmarked. The main memory requirement
will depend on vocabulary size and sparse matrix dimensions.

#### Software

The planned stack is Python 3.11, scikit-learn, joblib, MLflow for experiment
tracking, DVC for large artifacts, and CodeCarbon for energy tracking. The
repository's current starter code does not yet implement this pipeline, final
dependency versions must be pinned before release.

## Citation

There is no project-specific paper for this baseline. Cite the implementation
references and the dataset when reusing it:

- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [Shell Safety v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1)

## Glossary

- **ALLOW:** The project label for a command that may proceed without extra confirmation.
- **ASK:** The project label for a command that requires user confirmation.
- **DENY:** The project label for a command that should not be executed automatically.
- **TF-IDF:** Term-frequency/inverse-document-frequency weighting applied here to character n-grams.
- **Linear SVM:** A margin-based classifier whose decision function is linear in the sparse feature space.
- **Target leakage:** Use of information unavailable at prediction time or too closely derived from the target.

## More Information

See the [documentation overview](../index.md) for the shared project context.
The trained artifact, configuration, metrics, and checksum should be linked
here when they are created.

## Model Card Authors

SafeAgent project team, TAED2 course project.

## Model Card Contact

Use the issue tracker of the [taed2-SafeAgent GitHub repository](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/issues).
