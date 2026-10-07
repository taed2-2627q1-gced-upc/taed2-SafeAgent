import hashlib
import json
from pathlib import Path
import subprocess

import joblib

from taed2_safeagent.data.common import LABELS, load_params, read_jsonl
from taed2_safeagent.data.inputs import build_input, example_id, safe_context


def load_config(path):
    path = Path(path).resolve()
    params = load_params(path)
    settings = params["baseline"]
    if settings["input_mode"] != "command":
        raise ValueError("This baseline uses command inputs only")
    for field, folder in (("model_dir", "models"), ("report_dir", "reports")):
        target = (path.parent / settings[field]).resolve()
        parent = path.parent / folder
        if not target.is_relative_to(parent) or target == parent:
            raise ValueError(f"Invalid baseline output path: {field}")
        settings[field] = target
    return params


def load_examples(params, split):
    if split not in {"train", "validation"}:
        raise ValueError("Test is reserved for final evaluation")
    path = params["paths"]["processed"] / f"{split}.jsonl"
    rows = [row for _, row in read_jsonl(path)]
    if not rows:
        raise ValueError(f"Empty prepared partition: {split}")
    for row in rows:
        if set(row) != {"id", "group_id", "command", "context", "label"}:
            raise ValueError("Unexpected prepared fields")
        build_input(row, "command")
        if row["label"] not in LABELS:
            raise ValueError("Unknown target label")
        if not isinstance(row["group_id"], str) or not row["group_id"]:
            raise ValueError("Missing group identifier")
        if row["context"] != safe_context(row["context"]):
            raise ValueError("Context is not canonical")
        if row["id"] != example_id(row["command"], row["context"]):
            raise ValueError("Prepared identifier differs from the input")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Repeated prepared identifiers")
    if {row["label"] for row in rows} != set(LABELS):
        raise ValueError("Each partition must contain every class")
    return rows


def dataset_identity(params):
    path = params["paths"]["processed"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source = {key: params["source"][key] for key in ("repository", "revision")}
    protocol = {key: params[key] for key in ("context", "grouping", "split")}
    if manifest["source"] != source or manifest["protocol"] != protocol:
        raise ValueError("Prepared dataset differs from its configuration")
    if manifest["labels"] != list(LABELS):
        raise ValueError("Prepared label mapping differs")
    train_path = params["paths"]["processed"] / "train.jsonl"
    return {
        "source": source,
        "protocol": protocol,
        "train_content": hashlib.sha256(train_path.read_bytes()).hexdigest(),
    }


def code_identity():
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    )
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def load_model(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["input_mode"] != "command" or metadata["labels"] != list(LABELS):
        raise ValueError("Unexpected model input mode or labels")
    pipeline = joblib.load(directory / "pipeline.joblib")
    if pipeline.classes_.tolist() != list(LABELS):
        raise ValueError("Stored model classes differ from its metadata")
    return pipeline, metadata
