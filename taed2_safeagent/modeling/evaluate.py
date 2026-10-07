from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import typer

from taed2_safeagent.data.common import LABELS, write_json, write_jsonl
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import (
    dataset_identity,
    load_config,
    load_examples,
    load_model,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


def compute_metrics(expected, predicted):
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected aligned nonempty labels and predictions")
    if set(expected) - set(LABELS) or set(predicted) - set(LABELS):
        raise ValueError("Unknown evaluation labels")
    denied = sum(label == "DENY" for label in expected)
    false_allow = sum(
        actual == "DENY" and guess == "ALLOW" for actual, guess in zip(expected, predicted)
    )
    report = classification_report(
        expected, predicted, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "classes": {label: report[label] for label in LABELS},
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows": "true",
            "columns": "predicted",
            "counts": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
        },
        "deny_to_allow_count": false_allow,
        "deny_to_allow_rate": false_allow / denied if denied else None,
    }


def evaluate(params):
    report_dir = params["baseline"]["report_dir"]
    for filename in ("metrics.json", "predictions.jsonl"):
        (report_dir / filename).unlink(missing_ok=True)
    pipeline, metadata = load_model(params["baseline"]["model_dir"])
    if dataset_identity(params) != metadata["dataset"]:
        raise ValueError("Evaluation dataset differs from the training dataset")
    training = load_examples(params, "train")
    rows = load_examples(params, "validation")
    for field in ("id", "group_id"):
        if {row[field] for row in training} & {row[field] for row in rows}:
            raise ValueError(f"Training and validation overlap: {field}")
    predictions = pipeline.predict([build_input(row, "command") for row in rows]).tolist()
    metrics = {
        "split": "validation",
        "rows": len(rows),
        "model": metadata["model"],
        "input_mode": metadata["input_mode"],
        "settings": metadata["settings"],
        **compute_metrics([row["label"] for row in rows], predictions),
    }
    write_json(report_dir / "metrics.json", metrics)
    write_jsonl(
        report_dir / "predictions.jsonl",
        [
            {"id": row["id"], "true_label": row["label"], "predicted_label": prediction}
            for row, prediction in zip(rows, predictions)
        ],
    )
    return metrics


@app.command()
def main(params: Path = Path("params.yaml")):
    try:
        result = evaluate(load_config(params))
    except (ValueError, TypeError, FileNotFoundError) as error:
        typer.echo(f"Evaluation failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Validation macro F1: {result['macro_f1']:.3f}")
    typer.echo(f"DENY recall: {result['classes']['DENY']['recall']:.3f}")
    typer.echo(f"DENY predicted ALLOW: {result['deny_to_allow_count']}")


if __name__ == "__main__":
    app()
