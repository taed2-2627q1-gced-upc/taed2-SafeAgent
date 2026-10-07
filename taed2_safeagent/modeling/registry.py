import json
import os
from pathlib import Path

from dvc.repo import Repo
import joblib
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from taed2_safeagent.data.common import LABELS, STATUS_FIELDS
from taed2_safeagent.data.inputs import build_input, safe_context


def input_texts(data, mode):
    if "command" not in data or set(data.columns) - {"command", "context"}:
        raise ValueError("Use command and optional context columns")
    texts = []
    for row in data.to_dict("records"):
        context = row.get("context")
        if isinstance(context, str):
            context = json.loads(context)
        context = {
            "gitRemote": None,
            "gitStatus": {field: [] for field in STATUS_FIELDS},
            "agentTouchedFiles": [],
            **(context or {}),
        }
        texts.append(
            build_input({"command": row["command"], "context": safe_context(context)}, mode)
        )
    return texts


class RegistryModel:
    """Load a saved model for registry predictions."""

    def __init__(self, directory, mode):
        self.mode = mode
        self.directory = Path(directory)
        self.encoder = not (self.directory / "pipeline.joblib").exists()
        if self.encoder:
            self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                directory, local_files_only=True, attn_implementation="sdpa"
            ).eval()
        else:
            self.model = joblib.load(self.directory / "pipeline.joblib")

    def predict(self, data, params=None):
        del params
        texts = input_texts(data, self.mode)
        if self.encoder:
            metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
            batch = self.tokenizer(
                texts,
                truncation=True,
                max_length=metadata["settings"]["max_length"],
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**batch).logits
            if not torch.isfinite(logits).all():
                raise ValueError("Invalid model scores")
            labels = [self.labels[index] for index in logits.argmax(dim=-1).tolist()]
        else:
            labels = self.model.predict(texts).tolist()
        return pd.DataFrame({"label": labels})

    @property
    def labels(self):
        return list(LABELS)


def _load_pyfunc(data_path):
    manifest = json.loads(Path(data_path).read_text(encoding="utf-8"))
    base = os.environ.get("SAFEAGENT_MODEL_CACHE", str(Path.home() / ".cache/safeagent/models"))
    directory = Path(base) / manifest["model_version"]
    if not (directory / "metadata.json").exists():
        token = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if not token:
            raise ValueError("Set the DagsHub token before recovering the model")
        remote = {
            "url": "s3://dvc",
            "endpointurl": "https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.s3",
            "access_key_id": token,
            "secret_access_key": token,
        }
        directory.parent.mkdir(parents=True, exist_ok=True)
        Repo.get(
            manifest["repository"],
            manifest["path"],
            out=str(directory),
            rev=manifest["output_commit"],
            remote="storage",
            remote_config=remote,
        )
    return RegistryModel(directory, manifest["input_mode"])
