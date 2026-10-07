---
model_id: taed2-safeagent/modernbert-base-shell-safety
language:
  - en
library_name:
  - transformers
  - pytorch
pipeline_tag: text-classification
tags:
  - shell-command-safety
  - command-risk-classification
  - modernbert
  - execution-context
  - draft
dataset:
  - tomngdev/shell-safety-v1.1
base_model: answerdotai/ModernBERT-base
status: draft
license: apache-2.0
license_name: Apache 2.0 for the ModernBERT backbone, project artifact terms pending
---

# Model Card for ModernBERT-base Shell Safety Classifier

The SafeAgent ModernBERT classifier predicts ALLOW, ASK or DENY for shell
commands. Command and command with context variants have been fitted on
the same prepared data, keeping the backbone and split fixed for comparison.
The validation results and saved artifacts are linked below.

## Model Details

### Model Description

ModernBERT-base is a bidirectional encoder-only Transformer with long-context
support and pretraining on English and code. The SafeAgent adaptation adds a
three-class classification head and is intended to consume a versioned text
serialization of the proposed command and the surrounding execution context.
It is a safety-classification component, not an agent, shell interpreter, or
execution sandbox.

- **Developed by:** SafeAgent project team, TAED2 course project
- **Funded by:** Academic course project, no external funding identified
- **Shared by:** SafeAgent project team
- **Model type:** ModernBERT encoder with a three-class sequence-classification head
- **Language(s):** English shell commands and execution-context text
- **License:** The ModernBERT backbone is released under Apache 2.0. The license and redistribution terms for the project fine-tuned artifact are still to be declared, dataset and dependency terms also apply.
- **Finetuned from model:** [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

### Model Sources

- **Project repository:** [taed2-SafeAgent on GitHub](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent)
- **Base model:** [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- **Fine-tuning dataset:** [tomngdev/shell-safety-v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1)
- **Paper:** [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)
- **Demo:** Not available

## Uses

### Direct Use

Given a proposed shell command and its available session context, the fine-tuned
classifier returns `ALLOW`, `ASK`, or `DENY`. The input text is only analyzed,
the model must not execute it or treat the prediction as a complete security
decision.

### Downstream Use

The intended downstream use is as a contextual risk signal inside an AI coding
agent's approval workflow. It may support comparisons against the command-only
CodeBERT and TF-IDF models, or be served behind a validated API after latency,
failure handling, policy integration, and safety testing are complete.

### Out-of-Scope Use

Do not use this model as a sole authorization decision, sandbox, malware
detector, command validator, or guarantee of safe execution. It is not intended
for arbitrary languages, unseen shells, untrusted production automation, or
high-impact decisions without deterministic controls and human/system review.

## Bias, Risks, and Limitations

- The fine-tuning dataset is synthetic and may not capture the diversity,
  ambiguity, or adversarial behavior of real agent sessions.
- Context can improve or worsen a decision. Stale, missing, manipulated, or
  incorrectly serialized context may produce a confident but unsafe result.
- The command and context fields may contain repository names, paths, prompts,
  remotes, or other sensitive information. Apply appropriate access control and
  data minimization before logging or sending them to a service.
- ModernBERT's pretraining is primarily English and code. Performance may be
  lower for other languages, shell dialects, or operational conventions.
- Long-context capability does not guarantee reliable reasoning about shell
  side effects. Long inputs may also increase memory use and latency.
- False `ALLOW` predictions can enable harmful actions, false `ASK`/`DENY`
  predictions can interrupt legitimate work.
- `category` and `reason` are excluded from the predictive input because they
  can leak label semantics. They may be used for evaluation and error analysis.
- Scores are not calibrated probabilities unless calibration is measured on an
  untouched validation set.

### Recommendations

Version the exact context serialization and reject or quarantine missing,
malformed, or suspicious context. Prioritize `DENY` recall, route uncertain
cases to `ASK`, and fail closed on model/service errors. Combine the model with
least privilege, sandboxing, deterministic policy rules, human confirmation,
and tamper-evident audit logs. Evaluate context ablations with identical data
splits and training protocols. Monitor drift by shell, repository type,
command length, context length, and target class, and red-team context
injection, obfuscation, command chaining, and state-dependent behavior.

## How to Get Started with the Model

Recover the fitted model with DVC and install the frozen training group.
The artifact includes its tokenizer and explicit class mapping.

```python
from transformers import pipeline
from taed2_safeagent.data.inputs import build_input

model_path = "models/encoders/modernbert__command-context__2__a740d875"
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
example = {"command": "git status", "context": {}}
text = build_input(example, "command-context")
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

The primary ModernBERT experiment uses command and the allowed safe_context_v1
profile. Shell may be used only in a separate ablation or analysis. Training
must reference the fixed data protocol and its DVC version. Missing optional
context fields follow the null defaults in the shared input builder.

The ModernBERT base model was pretrained on approximately 2 trillion tokens of
English text and code. That pretraining corpus is distinct from the project's
SafeAgent fine-tuning data.

### Training Procedure

#### Preprocessing

Serialize the command and session context with explicit field boundaries. The
format must be stable, escaped where necessary, and versioned. Tokenize with the
ModernBERT tokenizer, define a maximum sequence length, and document truncation
priority so that the command is not silently discarded. Do not concatenate
`category` or `reason` to the input. If the command-only ablation is trained,
publish it under a separate model ID.

#### Training Hyperparameters

- **Training regime:** The shared trainer fine tunes the encoder and classification head in both input modes. Initial settings use two epochs, learning rate 0.00002, effective batch 32, seed 42 and a 512 token limit. GPU runs use mixed precision and validation selects the best epoch. The saved runs use these settings.

#### Speeds, Sizes, Times

The fitted classifier has 149,607,171 parameters. Inference
latency and peak memory have not been measured.

| Input | Fitting seconds | Model and tokenizer MiB |
| --- | ---: | ---: |
| command | 792.5 | 574.2 |
| command-context | 1354.9 | 574.2 |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Use a held-out, versioned split of the same dataset with duplicate and
near-duplicate leakage checks. The final card must record the dataset revision,
split construction, counts, class distribution, missing-context rate, and
serialization version.

#### Factors

Disaggregate by target class, shell family, command length, context length,
context availability, context completeness, command chaining/operators, and
`category` for analysis-only slices. Compare against a command-only model on the
same split when measuring the value of context. Do not change the input and
model family simultaneously for that ablation.

#### Metrics

Report `DENY` recall as the primary safety metric, plus per-class
precision/recall/F1, macro-F1, balanced accuracy, accuracy, confusion matrix,
and inference latency/memory. For context-specific behavior, report performance
on complete versus missing/stale-context slices. If scores route cases to
`ASK`, report calibration and coverage-risk curves.

### Results

Both modes use 26,660 training and 3,333 validation examples from the strict
group split. The 3,332 test examples remain reserved. These validation
scores on synthetic examples do not establish safety on live commands.

| Input | Macro F1 | DENY recall | DENY predicted ALLOW | Shared run |
| --- | ---: | ---: | ---: | --- |
| command | 0.8647 | 0.8547 | 22 | [MLflow](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/4851c0df192344fa99eacf1721bdbafe) |
| command-context | 0.8953 | 0.8678 | 14 | [MLflow](https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.mlflow/#/experiments/1/runs/5ed5c367a6c8481d93a744ce97e7a7b0) |

The linked runs include per class metrics and raw and normalized confusion
matrices. DVC recovers the weights and aligned validation predictions.

#### Summary

One seed and two epochs provide an initial paired comparison.
Final model selection also needs safety errors and serving checks.

## Model Examination

Planned examination includes error buckets by context quality, paired examples
where only context changes, counterfactual command edits, calibration plots,
and manual review of false `ALLOW` predictions. Attribution or attention views
are diagnostic aids only and should not be interpreted as causal explanations
or execution guarantees.

## Environmental Impact

CodeCarbon covered fitting and epoch validation on Kaggle using GPU 0.
GPU power monitoring was available, while CPU power used a constant
fallback estimate and RAM power is estimated. The USA carbon factor is
assumed because the physical region is unverified. The second allocated
GPU is excluded, so these estimates do not cover the full allocated machine.

| Input | Energy kWh | Emissions kg CO2e | Duration seconds |
| --- | ---: | ---: | ---: |
| command | 0.025860 | 0.009555 | 791.3 |
| command-context | 0.044334 | 0.016380 | 1353.7 |

## Technical Specifications

### Model Architecture and Objective

The fitted architecture is:

```text
command + session_context
    -> versioned field serialization
    -> ModernBERT tokenizer
    -> answerdotai/ModernBERT-base encoder
    -> three-class sequence-classification head
    -> ALLOW / ASK / DENY
```

The ModernBERT backbone is an encoder-only, pre-norm Transformer with GeGLU
activations, RoPE, and local-global alternating attention. It was pretrained
with a masked-language-model objective, the SafeAgent adaptation uses supervised
three-class classification. The final loss, class weights, truncation, and
decision policy must be stored with the fine-tuning configuration.

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
@misc{modernbert,
  title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
  author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
  year={2024},
  eprint={2412.13663},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.13663}
}
```

**APA:**

Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini,
S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G.,
Howard, J., & Poli, I. (2024). *Smarter, better, faster, longer: A modern
bidirectional encoder for fast, memory efficient, and long context finetuning
and inference*. arXiv:2412.13663.

Also cite the [ModernBERT model card](https://huggingface.co/answerdotai/ModernBERT-base)
and the [Shell Safety v1.1 dataset](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1)
when redistributing the complete project.

## Glossary

- **ALLOW / ASK / DENY:** SafeAgent's three operational-risk labels.
- **Session context:** State and metadata surrounding the proposed command, such as repository status and recent user/agent activity.
- **Context ablation:** A comparison that removes context while keeping the model family and evaluation split controlled.
- **RoPE:** Rotary positional embeddings, used by ModernBERT for position information.
- **DENY recall:** Fraction of truly denied examples that the model identifies as `DENY`.
- **Target leakage:** Predictive use of fields such as `reason` that expose or closely encode the label.

## More Information

The result table links the shared runs, while reports/encoders stores
their small metrics and run links. DVC pointers recover fitted models.

## Model Card Authors

SafeAgent project team, TAED2 course project.

## Model Card Contact

Use the issue tracker of the [taed2-SafeAgent GitHub repository](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/issues).
