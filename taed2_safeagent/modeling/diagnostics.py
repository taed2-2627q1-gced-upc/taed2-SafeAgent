from importlib.metadata import version
import json
from pathlib import Path

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ConfusionMatrixReport, SingleDatasetPerformance
import pandas as pd
import typer

from taed2_safeagent.data.common import LABELS, load_params, read_jsonl, write_json
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import load_examples

MODEL_RUNS = (
    ("codebert__command__2__a740d875", "command"),
    ("codebert__command-context__2__a740d875", "command-context"),
    ("modernbert__command__2__a740d875", "command"),
    ("modernbert__command-context__2__a740d875", "command-context"),
)

app = typer.Typer(pretty_exceptions_show_locals=False)


def _index_rows(rows, description):
    indexed = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Missing example ID in {description}")
        if identifier in indexed:
            raise ValueError(f"Duplicate example ID in {description}: {identifier}")
        indexed[identifier] = row
    return indexed


def align_validation_data(validation_rows, prediction_rows, joined_rows, model_name, input_mode):
    """Verify predictions and source metadata refer to the prepared validation examples."""
    examples = _index_rows(validation_rows, "prepared validation data")
    predictions = _index_rows(prediction_rows, f"{model_name} predictions")
    joined = _index_rows(joined_rows, f"{model_name} joined metadata")

    expected_ids = set(examples)
    for description, indexed in (("predictions", predictions), ("joined metadata", joined)):
        if set(indexed) != expected_ids:
            missing = len(expected_ids - set(indexed))
            extra = len(set(indexed) - expected_ids)
            raise ValueError(
                f"{model_name} {description} do not match validation IDs "
                f"(missing={missing}, extra={extra})"
            )

    for identifier, example in examples.items():
        prediction = predictions[identifier]
        metadata = joined[identifier]
        if not {"category", "shell"} <= metadata.keys():
            raise ValueError(f"Source metadata is missing for {model_name}: {identifier}")
        if set(prediction) != {"id", "true_label", "predicted_label"}:
            raise ValueError(f"Unexpected prediction fields for {model_name}: {identifier}")
        if prediction["true_label"] != example["label"]:
            raise ValueError(
                f"True label differs from validation data for {model_name}: {identifier}"
            )
        if prediction["predicted_label"] not in LABELS:
            raise ValueError(f"Unknown predicted label for {model_name}: {identifier}")
        for field in ("true_label", "predicted_label"):
            if metadata.get(field) != prediction[field]:
                raise ValueError(f"Joined {field} differs for {model_name}: {identifier}")

    return [
        {
            "id": example["id"],
            "text": build_input(example, input_mode),
            "true_label": example["label"],
            "predicted_label": predictions[example["id"]]["predicted_label"],
        }
        for example in validation_rows
    ]


def _table_result(value, check_name):
    if not isinstance(value, pd.DataFrame):
        raise TypeError(
            f"Unexpected result type from Deepchecks {check_name}: {type(value).__name__}"
        )
    rows = []
    for record in value.to_dict(orient="records"):
        rows.append(
            {
                str(column): (
                    None if pd.isna(item) else item.item() if hasattr(item, "item") else item
                )
                for column, item in record.items()
            }
        )
    return {
        "columns": [str(column) for column in value.columns],
        "index": [
            None if pd.isna(item) else item.item() if hasattr(item, "item") else item
            for item in value.index
        ],
        "rows": rows,
    }


def _run_model_diagnostics(examples, model_name, input_mode):
    frame = pd.DataFrame(
        {
            "text": [example["text"] for example in examples],
            "label": [example["true_label"] for example in examples],
        }
    )
    dataset = Dataset(
        frame,
        label="label",
        features=["text"],
        cat_features=[],
        label_type="multiclass",
    )
    predictions = [example["predicted_label"] for example in examples]
    checks = {
        "single_dataset_performance": SingleDatasetPerformance().run(
            dataset, y_pred=predictions, model_classes=list(LABELS), with_display=False
        ),
        "confusion_matrix": ConfusionMatrixReport(normalize_display=False).run(
            dataset, y_pred=predictions, model_classes=list(LABELS), with_display=False
        ),
    }
    return {
        "model": model_name,
        "input_mode": input_mode,
        "split": "validation",
        "rows": len(examples),
        "alignment": {
            "status": "passed",
            "ids_match_validation": True,
            "true_labels_match_validation": True,
            "joined_metadata_matches_predictions": True,
        },
        "deepchecks": {
            "version": version("deepchecks"),
            "checks": {
                name: {
                    "status": "completed",
                    "result": _table_result(result.value, name),
                }
                for name, result in checks.items()
            },
            "unavailable_checks": [
                {
                    "name": "probability-based diagnostics",
                    "reason": (
                        "Saved validation artifacts contain hard class predictions only; "
                        "class probabilities are unavailable."
                    ),
                }
            ],
        },
    }


def run_diagnostics(params_path=Path("params.yaml")):
    params_path = Path(params_path).resolve()
    root = params_path.parent
    output_paths = [
        root / "reports/encoders" / model_name / "diagnostics.json" for model_name, _ in MODEL_RUNS
    ]
    for output_path in output_paths:
        output_path.unlink(missing_ok=True)
    params = load_params(params_path)
    validation_path = params["paths"]["reports"] / "validation.json"
    validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation_report.get("success") is not True:
        raise ValueError("Run data validation successfully before model diagnostics")

    validation_rows = load_examples(params, "validation")
    reports = []
    for (model_name, input_mode), output_path in zip(MODEL_RUNS, output_paths):
        model_dir = root / "models/encoders" / model_name
        prediction_rows = [
            row for _, row in read_jsonl(model_dir / "validation_predictions.jsonl")
        ]
        joined_rows = [
            row for _, row in read_jsonl(model_dir / "predictions_joined_with_metadata.jsonl")
        ]
        aligned = align_validation_data(
            validation_rows, prediction_rows, joined_rows, model_name, input_mode
        )
        report = _run_model_diagnostics(aligned, model_name, input_mode)
        report["data_quality"] = {
            "status": "passed",
            "deepchecks": validation_report["deepchecks"],
            "great_expectations": validation_report["great_expectations"],
        }
        write_json(output_path, report)
        reports.append(output_path)
    return reports


@app.command()
def main(params: Path = Path("params.yaml")):
    try:
        reports = run_diagnostics(params)
    except (KeyError, OSError, TypeError, ValueError) as error:
        typer.echo(f"Model diagnostics failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Saved Deepchecks diagnostics for {len(reports)} models")


if __name__ == "__main__":
    app()
