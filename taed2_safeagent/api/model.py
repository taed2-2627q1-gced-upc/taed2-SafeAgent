from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from taed2_safeagent.api.schemas import ModelInfo
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import load_model

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models/tfidf_char_linear_svm"


@dataclass
class Predictor:
    """Use the existing baseline loader and input builder."""

    pipeline: Any
    info: ModelInfo

    def predict(self, command, context):
        text = build_input({"command": command, "context": context}, self.info.input_mode)
        return self.pipeline.predict([text])[0]


def load_predictor(directory=None):
    directory = Path(directory or os.environ.get("SAFEAGENT_MODEL_DIR", DEFAULT_MODEL_DIR))
    pipeline, metadata = load_model(directory)
    with (directory / "pipeline.joblib").open("rb") as stream:
        version = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    return Predictor(
        pipeline,
        ModelInfo(
            model=metadata["model"],
            model_version=version,
            input_mode=metadata["input_mode"],
        ),
    )
