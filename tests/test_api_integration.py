import hashlib

from fastapi.testclient import TestClient
import pytest

from taed2_safeagent.api.app import create_app
from taed2_safeagent.api.model import DEFAULT_MODEL_DIR
from taed2_safeagent.data.common import LABELS
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import load_model


@pytest.mark.integration
def test_http_predictions_match_the_saved_baseline(monkeypatch):
    if not (DEFAULT_MODEL_DIR / "metadata.json").is_file():
        pytest.skip("Recover models/tfidf_char_linear_svm with DVC first")
    monkeypatch.setenv("SAFEAGENT_MODEL_DIR", str(DEFAULT_MODEL_DIR))
    pipeline, metadata = load_model(DEFAULT_MODEL_DIR)
    artifact = DEFAULT_MODEL_DIR / "pipeline.joblib"
    original_bytes = artifact.read_bytes()
    version = "sha256:" + hashlib.sha256(original_bytes).hexdigest()
    commands = [
        "git status",
        "  git diff\n",
        "git push origin main",
        "rm -rf ./build",
        "echo café",
    ]
    expected = pipeline.predict([build_input({"command": text}, "command") for text in commands])
    with TestClient(create_app()) as connection:
        ready = connection.get("/ready")
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "model": metadata["model"],
            "model_version": version,
            "input_mode": "command",
        }
        for command, label in zip(commands, expected, strict=True):
            assert label in LABELS
            for context in (None, {"cwd": "/example", "gitStatus": {"modified": ["app.py"]}}):
                payload = {"command": command, "context": context}
                response = connection.post("/predict", json=payload)
                assert response.status_code == 200
                assert response.json() == {
                    "label": label,
                    "model": metadata["model"],
                    "model_version": version,
                    "input_mode": "command",
                }
    assert artifact.read_bytes() == original_bytes
