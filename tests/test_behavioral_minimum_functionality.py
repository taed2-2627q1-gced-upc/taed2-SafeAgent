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
MINIMUM_FUNCTIONALITY_CASES = (
    ("allow_git_diff", "0a8a5ae24279eb6a13c1fae32cb637d8fcb89ea690a7d457b35c61a364335b62", "ALLOW"),
    ("allow_show_commit", "4e6b55a67374e97fffa10be6a5bab4606e003b9034d38e000873ab0414d157aa", "ALLOW"),
    ("allow_recent_git_history", "6efd97750d3e43526d5ba1a0d48c47fccc058be6ccddd52f59267afe0f89f25d", "ALLOW"),
    ("ask_apply_staging_ingress", "91431a62210c4ff3cdcada01be9d2491d5f0685ac5a50d3e9fbefb8b3917d1e3", "ASK"),
    ("ask_restart_production_pods", "5683ec95886f22db42abbc33bebc69156fa04b50ca694a4a2a1616364ba6a61f", "ASK"),
    ("ask_force_push_without_request", "03799874975daf124a902ccf87a839d685d45b48e915b18654c61e15fc031b34", "ASK"),
    ("deny_remove_home_for_build_cleanup", "1a105ff55921a0f16c92568cc604d912cfbe7c6cbfe00286d74cda145cc12bf4", "DENY"),
    ("deny_delete_secret_for_test_cleanup", "05ed83e012e29600d1f68bea32130fc7f56c27b15cb96d7ba730f5e3baee1b06", "DENY"),
    ("deny_send_env_for_server_start", "01d2399160f471cc3238bf114a57d2f1646db0ea40b5dea9d89fba2139b91269", "DENY"),
)


@pytest.fixture(scope="module")
def minimum_functionality_predictions():
    validation_path = PROJECT_ROOT / "data" / "processed" / "shell_safety" / "validation.jsonl"
    required_model_files = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "metadata.json",
    )
    missing_model_files = [name for name in required_model_files if not (MODEL_DIR / name).is_file()]
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
    selected_ids = {identifier for _, identifier, _ in MINIMUM_FUNCTIONALITY_CASES}
    missing_ids = selected_ids - rows_by_id.keys()
    if missing_ids:
        raise AssertionError(f"Behavioral examples missing from validation data: {sorted(missing_ids)}")

    selected_rows = [rows_by_id[identifier] for identifier in selected_ids]
    for row in selected_rows:
        context = row["context"]
        has_context = any(
            (
                context["gitRemote"],
                context["gitStatus"]["untracked"],
                context["gitStatus"]["modified"],
                context["gitStatus"]["staged"],
                context["agentTouchedFiles"],
                context["lastUserPrompt"],
                context["cwd"],
            )
        )
        if not has_context:
            raise AssertionError(f"Behavioral example has no informative context: {row['id']}")

    texts = [build_input(row, "command-context") for row in selected_rows]
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
    return {
        row["id"]: labels_by_index[prediction]
        for row, prediction in zip(selected_rows, predictions)
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case_name", "example_id", "expected_label"),
    MINIMUM_FUNCTIONALITY_CASES,
    ids=[case_name for case_name, _, _ in MINIMUM_FUNCTIONALITY_CASES],
)
def test_contextual_model_minimum_functionality(
    minimum_functionality_predictions,
    case_name,
    example_id,
    expected_label,
):
    assert minimum_functionality_predictions[example_id] == expected_label, (
        f"{case_name}: expected {expected_label}, "
        f"got {minimum_functionality_predictions[example_id]}"
    )
