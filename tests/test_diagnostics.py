from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from taed2_safeagent.data.common import read_jsonl, write_jsonl
from taed2_safeagent.data.validation import validate
from taed2_safeagent.modeling.diagnostics import (
    MODEL_RUNS,
    align_validation_data,
    run_diagnostics,
)


def test_alignment_binds_text_labels_predictions_and_metadata():
    examples = [{"id": "a", "command": "ls", "context": {}, "label": "ALLOW"}]
    predictions = [{"id": "a", "true_label": "ALLOW", "predicted_label": "DENY"}]
    joined = [
        {
            "id": "a",
            "true_label": "ALLOW",
            "predicted_label": "DENY",
            "shell": "posix",
            "category": "filesystem",
        }
    ]

    aligned = align_validation_data(examples, predictions, joined, "model", "command")

    assert aligned == [
        {
            "id": "a",
            "text": "ls",
            "true_label": "ALLOW",
            "predicted_label": "DENY",
        }
    ]


@pytest.mark.parametrize(
    ("prediction_change", "joined_change", "message"),
    [
        ({"true_label": "DENY"}, {}, "True label differs"),
        ({"id": "other"}, {}, "do not match validation IDs"),
        ({}, {"predicted_label": "ASK"}, "Joined predicted_label differs"),
    ],
)
def test_alignment_rejects_mismatched_predictions_and_metadata(
    prediction_change, joined_change, message
):
    examples = [{"id": "a", "command": "ls", "context": {}, "label": "ALLOW"}]
    predictions = [{"id": "a", "true_label": "ALLOW", "predicted_label": "DENY"}]
    joined = [
        {
            "id": "a",
            "true_label": "ALLOW",
            "predicted_label": "DENY",
            "shell": "posix",
            "category": "filesystem",
        }
    ]
    predictions[0].update(prediction_change)
    joined[0].update(joined_change)

    with pytest.raises(ValueError, match=message):
        align_validation_data(examples, predictions, joined, "model", "command")


def test_run_diagnostics_saves_deepchecks_outputs(bundle, tmp_path):
    validation = validate(bundle)
    root = Path(tmp_path)
    rows = [row for _, row in read_jsonl(bundle["paths"]["processed"] / "validation.jsonl")]
    for model_name, _ in MODEL_RUNS:
        model_dir = root / "models/encoders" / model_name
        predictions = [
            {
                "id": row["id"],
                "true_label": row["label"],
                "predicted_label": row["label"],
            }
            for row in rows
        ]
        write_jsonl(model_dir / "validation_predictions.jsonl", predictions)
        write_jsonl(
            model_dir / "predictions_joined_with_metadata.jsonl",
            [
                {**prediction, "shell": "posix", "category": "filesystem"}
                for prediction in predictions
            ],
        )

    settings = deepcopy(bundle)
    settings["paths"] = {
        name: Path(path).relative_to(root).as_posix() for name, path in settings["paths"].items()
    }
    params_path = root / "params.yaml"
    params_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    output_paths = run_diagnostics(params_path)

    assert len(output_paths) == len(MODEL_RUNS)
    for output_path in output_paths:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["rows"] == len(rows)
        assert report["alignment"]["status"] == "passed"
        assert report["data_quality"]["status"] == "passed"
        assert report["data_quality"]["deepchecks"] == validation["deepchecks"]
        assert set(report["deepchecks"]["checks"]) == {
            "single_dataset_performance",
            "confusion_matrix",
        }
        assert all(
            check["status"] == "completed" for check in report["deepchecks"]["checks"].values()
        )
        assert report["deepchecks"]["checks"]["single_dataset_performance"]["result"]["rows"]
        assert report["deepchecks"]["checks"]["confusion_matrix"]["result"]["rows"]
        assert report["deepchecks"]["unavailable_checks"]
