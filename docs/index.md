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
limits. The baseline and both transformer families have validation results,
and the transformer cards include paired command and context runs from Kaggle.
See the [baseline guide](baseline.md) and [encoder guide](encoder-training.md)
to train or recover the models.

## Project setup

See the [getting started guide](getting-started.md) and the
[repository README](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent#readme).
See [feature ownership](feature-ownership.md) for team responsibilities and the
pull request review protocol.
