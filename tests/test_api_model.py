import hashlib
import json

import joblib
import pytest
from sklearn.dummy import DummyClassifier

from taed2_safeagent.api.model import load_predictor
from taed2_safeagent.data.common import LABELS


@pytest.fixture(name="saved_model")
def model_directory(tmp_path):
    pipeline = DummyClassifier(strategy="constant", constant="ASK")
    pipeline.fit(["one", "two", "three"], list(LABELS))
    joblib.dump(pipeline, tmp_path / "pipeline.joblib")
    metadata = {"model": "fixture", "input_mode": "command", "labels": list(LABELS)}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def test_loader_identifies_saved_bytes_and_uses_configured_directory(saved_model, monkeypatch):
    monkeypatch.setenv("SAFEAGENT_MODEL_DIR", str(saved_model))
    predictor = load_predictor()
    version = hashlib.sha256((saved_model / "pipeline.joblib").read_bytes()).hexdigest()
    assert predictor.info.model_dump() == {
        "model": "fixture",
        "model_version": "sha256:" + version,
        "input_mode": "command",
    }
    assert predictor.predict("  git status\n", None) == "ASK"


def test_explicit_directory_overrides_environment(saved_model, monkeypatch):
    monkeypatch.setenv("SAFEAGENT_MODEL_DIR", str(saved_model / "missing"))
    assert load_predictor(saved_model).info.model == "fixture"


@pytest.mark.parametrize(
    "field,value",
    [("labels", ["DENY", "ASK", "ALLOW"]), ("input_mode", "command-context"), ("model", "")],
)
def test_incompatible_model_metadata_is_rejected(saved_model, field, value):
    path = saved_model / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_predictor(saved_model)


def test_model_classes_must_match_metadata(saved_model):
    path = saved_model / "pipeline.joblib"
    pipeline = joblib.load(path)
    pipeline.classes_ = pipeline.classes_[::-1]
    joblib.dump(pipeline, path)
    with pytest.raises(ValueError, match="classes differ"):
        load_predictor(saved_model)
