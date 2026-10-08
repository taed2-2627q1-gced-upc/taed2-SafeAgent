# Feature ownership

## Purpose

Feature ownership defines who maintains each part of SafeAgent and who provides
an independent review before changes are merged. The primary owner coordinates
feature maintenance, ensures that changes include appropriate tests, keeps the
documentation up to date, and addresses pull request feedback. The secondary
reviewer provides an independent review, with particular attention to the
feature's contracts and risks.

These assignments reflect the repository's current state. The project follows
the Cookiecutter Data Science layout, with data versioned through DVC, code in
`taed2_safeagent/`, tests in `tests/`, and documentation in `docs/`. Status
describes observed implementation, not a delivery date.

## Assignment summary

| Feature / module | Key components / files | Primary owner | Reviewer | Status / milestone |
| --- | --- | --- | --- | --- |
| Pinned dataset ingestion and source audit | `data/raw/shell_safety.dvc`, `taed2_safeagent/data/source.py`, `taed2_safeagent/data/audit.py`, `params.yaml` (`source`), `tests/test_source.py`, `tests/test_audit.py` | David González | Joel Márquez | Implemented: pinned snapshot, hash verification, and audit |
| Grouping, split, and data leakage prevention | `taed2_safeagent/data/grouping.py`, `taed2_safeagent/data/split.py`, `params.yaml` (`grouping`, `split`), `tests/test_grouping.py` | Joel Márquez | Pau González | Implemented: group-based train/validation/test split |
| Data contracts and validation | `taed2_safeagent/data/inputs.py`, `taed2_safeagent/data/validation.py`, `taed2_safeagent/dataset.py`, `reports/data/`, `tests/test_validation.py` | Pau González | Pau Balaguer | Implemented: row, label, duplicate, and quarantine checks |
| Feature extraction and execution context | `taed2_safeagent/features.py`, `taed2_safeagent/data/inputs.py`, `params.yaml` (`context`, `baseline.tfidf`), `tests/test_baseline.py` | Pau Balaguer | David González | Implemented: character n-gram TF-IDF and `command-context` input; word n-grams are not currently present |
| Baseline modeling and DVC pipeline | `taed2_safeagent/modeling/train.py`, `taed2_safeagent/modeling/evaluate.py`, `dvc.yaml`, `dvc.lock`, `params.yaml` (`baseline`), `tests/test_baseline.py`, `tests/test_model_pipeline.py` | David González | Pau González | Implemented: linear SVM; Logistic Regression is not part of the current baseline |
| Experiments, encoders, and tracking | `taed2_safeagent/modeling/encoder.py`, `experiment.py`, `tracking.py`, `register.py`, `params.yaml` (`tracking`, `encoders`), `docs/encoder-training.md`, `tests/test_encoder.py`, `tests/test_tracking.py`, `tests/test_registration.py` | Pau Balaguer | Joel Márquez | Implemented: MLflow on DagsHub, encoder training and registration |
| Quality assurance and sustainability | `tests/`, `notebooks/kaggle_baseline.ipynb`, `.github/workflows/data.yml`, `pyproject.toml`, `Makefile`, `taed2_safeagent/modeling/energy.py`, energy reports in `reports/` | Joel Márquez | David González | Partially implemented: pytest, Ruff, Pylint, CI, and CodeCarbon; PyNBLint is a dependency but is not run in the workflow |
| Serving and deployment | Future design for `taed2_safeagent/` and deployment documentation | Pau González | Pau Balaguer | Pending: no FastAPI API, Pydantic schemas, ALLOW/ASK/DENY inference endpoints, or Virtech deployment configuration are present in the current code |

A dependency listed in `uv.lock` is not, by itself, evidence that a feature has
been integrated. Status is based on the code, project configuration, pipeline,
and operational documentation. In particular, `params.yaml` currently pins
`tomngdev/shell-safety-v1.1`; it is not configured to use `shell-safety-v2`.

## Responsibilities by team member

### David González

- **Data ingestion and audit:** Maintain reproducible retrieval of the pinned
  snapshot, its hashes and counts, provenance, deduplication, and quarantine.
  For the report, present the source identity, audit results, and known dataset
  limitations.
- **Baseline and DVC pipeline:** Maintain baseline training and evaluation and
  the dependencies of their stages. For the report, record parameters,
  validation metrics, relevant errors — especially `DENY` predictions labeled
  as `ALLOW` — and the data and model versions.

### Joel Márquez

- **Grouping, split, and leakage prevention:** Maintain fingerprinting,
  similarity, and group assignment rules; verify that groups do not cross
  partitions. For the report, explain the protocol, seed, ratios, and split
  diagnostics.
- **Quality assurance and sustainability:** Maintain tests, linting, CI, and
  available CodeCarbon measurements. For the report, document which checks ran,
  their results, CI coverage, and the limitations and uncertainty of energy
  estimates.

### Pau Balaguer

- **Feature extraction and context:** Maintain character vectorization and the
  `command` and `command-context` input builders. For the report, describe
  preprocessing and compare the input modes; identify any word n-gram
  experiment as an extension, not as current behavior.
- **Experiments, encoders, and tracking:** Maintain experiment reproducibility,
  metadata, and registration. For the report, compare models and input modes
  using the same split, and preserve the provenance of code, data, and results.

### Pau González

- **Data contracts and validation:** Maintain validation of prepared examples
  and their reports. For the report, explain invariants, expected counts,
  duplicate and quarantined rows, and contract limitations.
- **Serving and deployment:** Lead the future API and deployment design. Before
  marking this feature as implemented, define schemas, request/response
  validation, safe behavior for `ALLOW`/`ASK`/`DENY`, inference tests, and a
  verifiable deployment strategy for Virtech. Do not assume these components
  already exist.

## Pull request and ownership protocol

### Seven commit rules

The repository does not contain a previously documented list of seven commit
rules. To avoid attributing undocumented rules to the project, this section
formalizes a protocol aligned with the commit history, configuration, and
current reproduction practices:

1. Use descriptive branch names: `feature/<name>` for functional changes and
   `docs/<name>` for documentation changes.
2. Keep each commit focused on one logical change; do not mix unrelated feature
   work with cleanup or unrelated results.
3. Use Conventional Commit-style messages, for example `feat(model): ...`,
   `fix(data): ...`, `test: ...`, `docs: ...`, or `chore: ...`.
4. Never add credentials, tokens, or secrets to the repository or commit
   messages; use environment variables or Git-ignored local configuration.
5. Do not manually modify the pinned raw source. A source update requires
   explicit changes to the revision, hashes, counts, and data protocol.
6. Version large artifacts with DVC and review changes to `dvc.lock` and DVC
   pointers; keep the small reports intended for Git under version control.
7. Run the relevant tests and checks before requesting a merge. For code
   changes, run pytest, Ruff, and Pylint; for documentation changes, run
   `mkdocs build --strict`.

### Review and merge

- Every pull request must identify the feature and link to the relevant
  documentation and tests.
- For pull requests from `feature/<name>` branches, the primary owner of the
  affected feature must approve before merge. The secondary reviewer assigned
  in the table provides the second review. If the owner is the pull request
  author, another team member with responsibility for the feature must approve;
  agree on the reviewer in the pull request.
- Do not approve a change that weakens leakage, validation, or inference safety
  controls unless the new behavior is explicitly documented and tested.
- This table defines the team's agreement. GitHub will enforce approval
  automatically only if a branch protection rule is configured.
