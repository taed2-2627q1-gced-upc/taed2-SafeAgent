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
  - tomngdev/shell-safety-v2
base_model: microsoft/codebert-base
status: draft
license: other
license_name: Project artifact license to be defined, backbone terms apply
---

# Model Card for CodeBERT-base Shell Safety Classifier

This is the second initial SafeAgent model. It is planned as a three-class
sequence classifier obtained by fine-tuning `microsoft/codebert-base` on shell
commands from `tomngdev/shell-safety-v2`. The primary comparison uses the
`command` field only, so the effect of a code-pretrained encoder can be
compared fairly with the TF-IDF baseline. No fine-tuned checkpoint or project
evaluation result is currently available.

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
- **Fine-tuning dataset:** [tomngdev/shell-safety-v2](https://huggingface.co/datasets/tomngdev/shell-safety-v2)
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

No SafeAgent fine-tuned checkpoint is currently published. After a checkpoint
has been trained and exported, an illustrative inference pattern is:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="models/codebert_shell_safety",
    tokenizer="models/codebert_shell_safety",
)
result = classifier("sudo apt-get remove openssh-server")[0]
print(result)  # label and score, do not execute the command automatically
```

The final artifact must include an explicit mapping from model label IDs to
`ALLOW`, `ASK`, and `DENY`, plus the tokenizer and preprocessing configuration.

## Training Details

### Training Data

The project fine-tuning source is [Shell Safety v2](https://huggingface.co/datasets/tomngdev/shell-safety-v2), a synthetic dataset with 34,007 examples according to its dataset card. Relevant fields are `command`, `session_context`, `label`, `category`, `shell`, and `reason`. The target is `label`, normalized to the uppercase classes `ALLOW`, `ASK`, and `DENY`.

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

- **Training regime:** Planned supervised fine-tuning of the encoder and classification head, exact learning rate, batch size, number of epochs, warm-up, weight decay, seed, maximum length, and mixed-precision setting are pending the first reproducible run

#### Speeds, Sizes, Times

Not measured for the SafeAgent fine-tune. The base model is approximately
125M parameters according to the project planning description, final parameter
count, checkpoint size, peak memory, throughput, latency, and training time
must be recorded from the actual artifact and hardware.

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

No SafeAgent fine-tuning or evaluation run has been committed yet. Results are
**pending**, the base CodeBERT paper's results on its own NL-PL tasks must not
be presented as results for shell-command safety.

#### Summary

This model tests whether a code-pretrained encoder improves over the lexical
baseline while keeping the primary input fixed to `command`. Selection of a
final production candidate must consider safety, latency, size, cost, energy,
and operational reliability.

## Model Examination

Planned examination includes error buckets, counterfactual command edits,
token/attention diagnostics used cautiously, and manual review of false
`ALLOW` predictions. Any explanation is an aid to investigation, not a proof
of safety or a substitute for executing only inside independent controls.

## Environmental Impact

The SafeAgent fine-tuning run has not yet been measured. Training is expected
to use a Kaggle GPU workflow when available, CodeCarbon should be enabled for
the actual run.

- **Hardware Type:** Not measured, GPU expected for fine-tuning
- **Hours used:** Not measured
- **Cloud Provider:** Expected Kaggle, not yet confirmed for a run
- **Compute Region:** Not recorded
- **Carbon Emitted:** Not measured

## Technical Specifications

### Model Architecture and Objective

The project model is planned as:

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

Fine-tuning is expected to run on a Kaggle GPU or comparable accelerator. Mixed
precision may be used if numerically validated. Inference should be benchmarked
on the intended deployment CPU as well as the training GPU because operational
latency may differ substantially.

#### Hardware

Exact GPU model, VRAM, CPU, RAM, training duration, and peak memory are pending
the first run.

#### Software

The planned stack includes Python 3.11, PyTorch, Hugging Face Transformers,
Datasets, MLflow, DVC, and CodeCarbon. Exact versions, tokenizer revision,
CUDA/runtime information, seed, and training script commit must be recorded for
reproducibility. The current repository contains starter files but not yet a
CodeBERT training implementation.

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

Also cite the [Shell Safety v2 dataset](https://huggingface.co/datasets/tomngdev/shell-safety-v2)
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

See the [documentation overview](../index.md) for the shared project context.
Link the final checkpoint, DVC pointer, MLflow run, and evaluation report here
once they exist.

## Model Card Authors

SafeAgent project team, TAED2 course project.

## Model Card Contact

Use the issue tracker of the [taed2-SafeAgent GitHub repository](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/issues).
