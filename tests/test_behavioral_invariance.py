import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from taed2_safeagent.data.common import LABELS, load_params
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import load_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "modernbert__command-context__2__a740d875"
MODEL_DIR = PROJECT_ROOT / "models" / "encoders" / MODEL_NAME
INVARIANCE_CASES = (
    (
        "allow_read_config",
        "004ce1ea5db5148d4606fc148831833a4123ae71f7d2b8eae8e09e724b0b5949",
        "ALLOW",
    ),
    (
        "allow_python_version",
        "01fc63fe52c29db8f1a41fb08860b0c9c78e8ecc581217ad8b834143397d0055",
        "ALLOW",
    ),
    (
        "ask_fetch_api_config",
        "007c84678056ba7d0b150f50ddea55629cf523fa2d308acc9780f3af984fd49a",
        "ASK",
    ),
    (
        "ask_test_production_site",
        "00843532a9127530a8320ce069b364e0d7e2ce56e5263cc99648fbd10ad3696c",
        "ASK",
    ),
    (
        "deny_publish_secret",
        "00cbb4d433625f01dd2c13e6724925c59298437250975e1f0d05a14c4736859d",
        "DENY",
    ),
    (
        "deny_disable_firewall",
        "01503195c56875fca7ca1d82282049a526da92a65863158f9521ff3df42b2cd6",
        "DENY",
    ),
)
UNRELATED_CONTEXT_MARKER = "invariance-unrelated-marker.txt"


@pytest.fixture(scope="module")
def invariance_predictions():
    validation_path = PROJECT_ROOT / "data" / "processed" / "shell_safety" / "validation.jsonl"
    required_model_files = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "metadata.json",
    )
    missing_model_files = [
        name for name in required_model_files if not (MODEL_DIR / name).is_file()
    ]
    if missing_model_files or not validation_path.is_file():
        pytest.skip(
            "Behavioral inference requires prepared validation data and the local "
            f"{MODEL_NAME} DVC artifact"
        )

    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model"] != "modernbert" or metadata["input_mode"] != "command-context":
        raise ValueError(f"Unexpected model artifact: {MODEL_DIR}")
    if metadata["labels"] != list(LABELS):
        raise ValueError("Model label order differs from the project label order")

    params = load_params(PROJECT_ROOT / "params.yaml")
    validation_rows = load_examples(params, "validation")
    rows_by_id = {row["id"]: row for row in validation_rows}
    missing_ids = {identifier for _, identifier, _ in INVARIANCE_CASES} - rows_by_id.keys()
    if missing_ids:
        raise AssertionError(
            f"Behavioral examples missing from validation data: {sorted(missing_ids)}"
        )

    original_rows = [rows_by_id[identifier] for _, identifier, _ in INVARIANCE_CASES]
    altered_rows = []
    for row in original_rows:
        altered = copy.deepcopy(row)
        untracked = altered["context"]["gitStatus"]["untracked"]
        if UNRELATED_CONTEXT_MARKER in untracked:
            raise ValueError(f"Context marker already exists in example: {row['id']}")
        untracked.append(UNRELATED_CONTEXT_MARKER)
        untracked.sort()
        altered_rows.append(altered)

    rows = original_rows + altered_rows
    texts = [build_input(row, "command-context") for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.to("cpu")
    model.eval()
    torch.set_num_threads(1)
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=metadata["settings"]["max_length"],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
    predictions = logits.argmax(dim=-1).tolist()
    labels_by_index = [model.config.id2label[index] for index in range(len(LABELS))]
    if labels_by_index != list(LABELS):
        raise ValueError("Model configuration label mapping differs from metadata")
    predicted_labels = [labels_by_index[index] for index in predictions]
    return {
        case_name: (predicted_labels[index], predicted_labels[index + len(INVARIANCE_CASES)])
        for index, (case_name, _, _) in enumerate(INVARIANCE_CASES)
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case_name", "example_id", "expected_label"),
    INVARIANCE_CASES,
    ids=[case_name for case_name, _, _ in INVARIANCE_CASES],
)
def test_contextual_model_is_invariant_to_unrelated_untracked_file(
    invariance_predictions,
    case_name,
    example_id,
    expected_label,
):
    original_prediction, altered_prediction = invariance_predictions[case_name]
    assert original_prediction == expected_label, (
        f"{case_name} ({example_id}): expected original prediction {expected_label}, "
        f"got {original_prediction}"
    )
    assert altered_prediction == expected_label, (
        f"{case_name} ({example_id}): expected prediction {expected_label} after adding "
        f"{UNRELATED_CONTEXT_MARKER!r} to untracked files, got {altered_prediction}"
    )
