# SafeAgent documentation

SafeAgent classifies shell commands proposed by AI coding agents as `ALLOW`,
`ASK`, or `DENY`. The project compares command-only models with a model that
also uses execution context.

## Data

The [Dataset Card](dataset-card.md) explains the source audit, strict group split
and shared input formats. See [getting started](getting-started.md) for data
recovery, DVC reproduction and quality checks.

## Model cards

The [model cards](model-cards/index.md) document the three classifiers and their
limits. The first TF-IDF and SVM baseline has validation results, while the
transformer cards still describe planned work. See the [baseline guide](baseline.md)
to train or recover the first model and evaluate it.

## Project setup

See the [getting started guide](getting-started.md) and the
[repository README](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent#readme).
The [Kaggle guide](kaggle-training.md) explains notebook setup, secrets and
reproducible baseline execution before later GPU experiments.
