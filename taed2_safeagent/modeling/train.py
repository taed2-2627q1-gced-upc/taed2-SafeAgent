from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import warnings

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import typer

from taed2_safeagent.data.common import LABELS, write_json
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.features import make_vectorizer
from taed2_safeagent.modeling.common import (
    code_identity,
    dataset_identity,
    load_config,
    load_examples,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


def fit_pipeline(texts, labels, settings):
    pipeline = Pipeline(
        [
            ("tfidf", make_vectorizer(settings["tfidf"])),
            ("svm", LinearSVC(**settings["svm"])),
        ]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(texts, labels)
    except ConvergenceWarning as error:
        raise ValueError("The SVM did not converge") from error
    return pipeline


def train(params):
    rows = load_examples(params, "train")
    identity = dataset_identity(params)
    settings = params["baseline"]
    texts = [build_input(row, "command") for row in rows]
    labels = [row["label"] for row in rows]
    started = perf_counter()
    pipeline = fit_pipeline(texts, labels, settings)
    elapsed = perf_counter() - started
    metadata = {
        "model": "tfidf_char_linear_svm",
        "input_mode": "command",
        "labels": list(LABELS),
        "settings": {key: settings[key] for key in ("tfidf", "svm")},
        "dataset": identity,
        "code": code_identity(),
        "train_rows": len(rows),
        "training_seconds": elapsed,
        "features": len(pipeline.named_steps["tfidf"].vocabulary_),
        "iterations": int(pipeline.named_steps["svm"].n_iter_),
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "joblib")},
    }
    directory = settings["model_dir"]
    (settings["report_dir"] / "mlflow_run.json").unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / "pipeline.joblib", compress=3)
    write_json(directory / "metadata.json", metadata)
    return metadata


@app.command()
def main(params: Path = Path("params.yaml")):
    try:
        result = train(load_config(params))
    except (ValueError, TypeError, FileNotFoundError) as error:
        typer.echo(f"Training failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Baseline trained on {result['train_rows']} examples")


if __name__ == "__main__":
    app()
