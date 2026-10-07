---
model_id: taed2-safeagent/codebert-base-shell-safety
language:
  - en
library_name:
  - transformers
  - pytorch
pipeline_tag: text-classification
tags:
  - shell-command-safety
  - command-risk-classification
  - codebert
  - draft
dataset:
  - tomngdev/shell-safety-v1.1
base_model: microsoft/codebert-base
status: draft
license: other
license_name: Project artifact license to be defined, backbone terms apply
---

# Model Card for CodeBERT-base Shell Safety Classifier

The SafeAgent CodeBERT classifier predicts ALLOW, ASK or DENY for shell
commands. Command and command with context variants have been fitted on
the same prepared data, keeping the backbone and split fixed for comparison.
The validation results and saved artifacts are linked below.

## Model Details

### Model Description

The model adapts CodeBERT's encoder representations to the SafeAgent decision
labels `ALLOW`, `ASK`, and `DENY`. CodeBERT is a RoBERTa-initialized encoder
pre-trained on bimodal natural-language/code data with an MLM+RTD objective.
The SafeAgent artifact is a new downstream classifier, not the original
CodeBERT model and not a command-executing agent.

- **Developed by:** SafeAgent project team, TAED2 course project
- **Funded by:** Academic course project, no external funding identified
- **Shared by:** SafeAgent project team
- **Model type:** CodeBERT/RoBERTa encoder with a three-class sequence-classification head
- **Language(s):** English shell commands and command-related text
- **License:** The project artifact license has not yet been declared. The CodeBERT repository publishes its code under MIT, verify the terms for all redistributed weights, data, and dependencies before release.
- **Finetuned from model:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)

### Model Sources

- **Project repository:** [taed2-SafeAgent on GitHub](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent)
- **Base model:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **Base-model source code:** [Microsoft CodeBERT repository](https://github.com/microsoft/CodeBERT)
- **Fine-tuning dataset:** [tomngdev/shell-safety-v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1)
- **Paper:** [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- **Demo:** Not available

## Uses

### Direct Use

Given a proposed shell command, the fine-tuned classifier returns `ALLOW`,
`ASK`, or `DENY`. The base model's generic feature-extraction usage is not the
SafeAgent task, a downstream classification checkpoint and its label mapping
are required. The prediction is advisory and must not execute the input.

### Downstream Use

The intended downstream use is as a risk signal in an AI coding-agent approval
workflow, or as a transformer comparison point against the classical baseline.
It may be wrapped by a service such as FastAPI after latency, failure behavior,
and safety policies have been validated.

### Out-of-Scope Use

Do not use this model as a standalone authorization mechanism, shell sandbox,
malware detector, incident-response decision maker, or guarantee of safe
execution. It is not intended for arbitrary programming languages, unseen
shells, production infrastructure without defense in depth, or decisions where
a false `ALLOW` could cause irreversible harm without human or system review.

## Bias, Risks, and Limitations

- The fine-tuning data is synthetic and may contain annotation rules or lexical
  artifacts that do not generalize to real agent sessions.
- CodeBERT was pre-trained for programming-language and natural-language tasks,
  not specifically for shell safety. Its pretraining distribution does not
  guarantee reliable understanding of shell semantics or operational context.
- The command-only primary input omits working directory, privileges,
  repository state, operating system, user intent, and recent agent actions.
- Shell commands can be compositional, stateful, obfuscated, or environment
  dependent. Token-level similarity is not equivalent to an execution-level
  safety analysis.
- False `ALLOW` predictions are safety-critical, false `ASK`/`DENY` predictions
  create friction or block legitimate work.
- Fine-tuning can overfit duplicate or near-duplicate examples. Dataset
  splitting and leakage checks must be reported.
- `category` and `reason` must remain out of the predictive input. They are
  useful for evaluation slices and error analysis only.
- The model's confidence scores are not automatically calibrated probabilities.

### Recommendations

Treat `DENY` recall as a primary guardrail and use a validation set to select
thresholds or an uncertainty policy. Route uncertain predictions to `ASK` and
fail closed if tokenization, model loading, or service health checks fail.
Combine the model with deterministic policy rules, sandboxing, least privilege,
human confirmation, and audit logs. Test separately across POSIX, PowerShell,
and CMD-like inputs, command-length ranges, context availability, and risk
categories. Record the exact base-model revision and fine-tuned checkpoint
checksum.

## How to Get Started with the Model

Recover the fitted model with DVC and install the frozen training group.
The artifact includes its tokenizer and explicit class mapping.

```python
from transformers import pipeline
from taed2_safeagent.data.inputs import build_input

model_path = "models/encoders/codebert__command__2__a740d875"
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
example = {"command": "git status", "context": {}}
text = build_input(example, "command")
result = classifier(text, truncation=True, max_length=512)[0]
print(result)
```

The scores have not been calibrated as safety probabilities. Commands
are analyzed without execution.

## Training Details

### Training Data

The data protocol is fixed in the [Dataset Card](../dataset-card.md).
Use revision fee89770c315d525ef2ee42adee6ef9725a7621e and the strict group
split with 26,660 train, 3,333 validation and 3,332 test rows. Use the shared
input builder and prepared IDs. Predictive text excludes reason, category,
assistant text, shell tags and source metadata. A context ablation must
keep the model family and selection protocol fixed.

The project fine-tuning source is [Shell Safety v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1), a synthetic dataset with 34,007 examples in the pinned source audit. Relevant fields are `command`, `session_context`, `label`, `category`, `shell`, and `reason`. The target is `label`, normalized to the uppercase classes `ALLOW`, `ASK`, and `DENY`.

The primary CodeBERT experiment uses `command` only. `session_context` is
reserved for a separately named context experiment, `category` and `reason` are
excluded from model inputs to reduce target leakage. The exact dataset revision,
split counts, duplicate policy, and class balance must be recorded at training
time.

The base CodeBERT model was trained by its authors on bimodal documentation and
code data from CodeSearchNet. That pretraining data is distinct from the
SafeAgent fine-tuning data and should not be described as the project's own
training set.

### Training Procedure

#### Preprocessing

Tokenize the raw command with the CodeBERT tokenizer and apply a documented
maximum sequence length. The truncation policy, padding strategy, label mapping,
and handling of empty or malformed commands must be versioned with the run.
No `category` or `reason` fields are concatenated to the input. If context is
added later, it must be a separate experiment with a distinct model ID.

#### Training Hyperparameters

- **Training regime:** The shared trainer fine tunes the encoder and classification head in both input modes. Initial settings use two epochs, learning rate 0.00002, effective batch 32, seed 42 and a 512 token limit. GPU runs use mixed precision and validation selects the best epoch. The saved runs use these settings.

#### Speeds, Sizes, Times

The fitted classifier has 124,647,939 parameters. Inference
latency and peak memory have not been measured.

| Input | Fitting seconds | Model and tokenizer MiB |
| --- | ---: | ---: |
| command | 480.0 | 480.1 |
| command-context | 835.3 | 480.1 |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Use a held-out, versioned split of the same dataset with duplicate and
near-duplicate leakage checks. The final card must record the source revision,
split construction, number of examples, and class distribution.

#### Factors

Disaggregate by target class, shell family, command length, command operators
or chaining, presence/absence of context, and `category` for analysis-only
slices. Compare the command-only model with a separately trained
command-plus-context variant, do not change both the model and input
representation when the goal is to isolate context value.

#### Metrics

Report `DENY` recall as the primary safety metric, together with per-class
precision/recall/F1, macro-F1, balanced accuracy, accuracy, confusion matrix,
and CPU/GPU inference latency. If the model is used to route uncertain cases
to `ASK`, also report calibration and coverage-risk curves.

### Results

Both modes use 26,660 training and 3,333 validation examples from the strict
group split. The 3,332 test examples remain reserved. These validation
scores on synthetic examples do not establish safety on live commands.

| Input | Macro F1 | DENY recall | DENY predicted ALLOW | Shared run |
| --- | ---: | ---: | ---: | --- |
| command | 0.8857 | 0.8442 | 18 | [MLflow](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/a81f878bbbff4019b032b389a0909c01) |
| command-context | 0.8977 | 0.8586 | 7 | [MLflow](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/c3532eb8a9c34146926bc9d2e10d047a) |

The linked runs include per class metrics and raw and normalized confusion
matrices. DVC recovers the weights and aligned validation predictions.

#### Summary

One seed and two epochs provide an initial paired comparison.
Final model selection also needs safety errors and serving checks.

## Model Examination

Planned examination includes error buckets, counterfactual command edits,
token/attention diagnostics used cautiously, and manual review of false
`ALLOW` predictions. Any explanation is an aid to investigation, not a proof
of safety or a substitute for executing only inside independent controls.

## Environmental Impact

CodeCarbon covered fitting and epoch validation on Kaggle using GPU 0.
GPU power monitoring was available, while CPU power used a constant
fallback estimate and RAM power is estimated. The USA carbon factor is
assumed because the physical region is unverified. The second allocated
GPU is excluded, so these estimates do not cover the full allocated machine.

| Input | Energy kWh | Emissions kg CO2e | Duration seconds |
| --- | ---: | ---: | ---: |
| command | 0.015644 | 0.005780 | 478.8 |
| command-context | 0.027262 | 0.010073 | 834.1 |

## Technical Specifications

### Model Architecture and Objective

The fitted architecture is:

```text
command string
    -> CodeBERT tokenizer
    -> microsoft/codebert-base encoder
    -> three-class sequence-classification head
    -> ALLOW / ASK / DENY
```

The base model is initialized from RoBERTa-base and was pre-trained with an
MLM+RTD objective. The SafeAgent objective is supervised three-class
classification, its loss, label weights, and decision policy must be recorded
with the fine-tuning configuration.

### Compute Infrastructure

Fitting used one visible Tesla T4 GPU on Kaggle with mixed precision and
PyTorch SDPA attention. ModernBERT compilation was disabled. Peak memory
and deployment inference latency remain unmeasured.

#### Software

The frozen environment uses Python 3.11, PyTorch 2.8.0,
Transformers 4.57.6, Accelerate 1.11.0 and CodeCarbon 3.2.2. The source,
backbone and data versions are saved with each run. Settings are in
params.yaml. See the [encoder guide](../encoder-training.md).

## Citation

**BibTeX:**

```bibtex
@misc{feng2020codebert,
  title={CodeBERT: A Pre-Trained Model for Programming and Natural Languages},
  author={Zhangyin Feng and Daya Guo and Duyu Tang and Nan Duan and Xiaocheng Feng and Ming Gong and Linjun Shou and Bing Qin and Ting Liu and Daxin Jiang and Ming Zhou},
  year={2020},
  eprint={2002.08155},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

**APA:**

Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B.,
Liu, T., Jiang, D., & Zhou, M. (2020). *CodeBERT: A pre-trained model for
programming and natural languages*. arXiv:2002.08155.

Also cite the [Shell Safety v1.1 dataset](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1)
and the [Microsoft CodeBERT repository](https://github.com/microsoft/CodeBERT)
when redistributing the complete project.

## Glossary

- **ALLOW / ASK / DENY:** SafeAgent's three operational-risk labels.
- **MLM:** Masked language modeling.
- **RTD:** Replaced-token detection, used in CodeBERT pretraining.
- **Fine-tuning:** Updating a pretrained encoder for the SafeAgent classification task.
- **Target leakage:** Predictive use of fields such as `reason` that expose or closely encode the label.
- **DENY recall:** Fraction of truly denied examples that the model identifies as `DENY`.

## More Information

The result table links the shared runs, while reports/encoders stores
their small metrics and run links. DVC pointers recover fitted models.

## Model Card Authors

SafeAgent project team, TAED2 course project.

## Model Card Contact

Use the issue tracker of the [taed2-SafeAgent GitHub repository](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/issues).
