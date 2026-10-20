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
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "metadata.json",
)


def load_validation_rows_by_id():
    params = load_params(PROJECT_ROOT / "params.yaml")
    validation_rows = load_examples(params, "validation")
    return {row["id"]: row for row in validation_rows}


def load_behavioral_model(model_dir=MODEL_DIR):
    validation_path = PROJECT_ROOT / "data" / "processed" / "shell_safety" / "validation.jsonl"
    missing_model_files = [
        name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()
    ]
    if missing_model_files or not validation_path.is_file():
        pytest.skip(
            "Behavioral inference requires prepared validation data and the local "
            f"{MODEL_NAME} DVC artifact"
        )

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model"] != "modernbert" or metadata["input_mode"] != "command-context":
        raise ValueError(f"Unexpected model artifact: {model_dir}")
    if metadata["labels"] != list(LABELS):
        raise ValueError("Model label order differs from the project label order")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model, metadata


def predict_behavioral_labels(rows, tokenizer, model, metadata):
    inputs = [build_input(row, "command-context") for row in rows]
    torch.set_num_threads(1)
    tokens = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=int(metadata["settings"]["max_length"]),
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**tokens).logits
    predictions = logits.argmax(dim=-1).tolist()
    labels_by_index = [model.config.id2label[index] for index in range(len(LABELS))]
    if labels_by_index != list(LABELS):
        raise ValueError("Model configuration label mapping differs from metadata")
    return [labels_by_index[index] for index in predictions]
