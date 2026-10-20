from dataclasses import dataclass, field

from fastapi.testclient import TestClient
import pytest

from taed2_safeagent.api.app import create_app
from taed2_safeagent.api.schemas import (
    MAX_COMMAND_CHARS,
    MAX_CONTEXT_CHARS,
    CommandContext,
    ModelInfo,
)
from taed2_safeagent.data.common import LABELS, canonical_json


@dataclass
class StubPredictor:
    """Control inference without loading model weights."""

    label: str = "ALLOW"
    info: ModelInfo = field(
        default_factory=lambda: ModelInfo(
            model="fixture",
            model_version="fixture-v1",
            input_mode="command",
        )
    )
    calls: list = field(default_factory=list)
    error: Exception | None = None

    def predict(self, command, context):
        self.calls.append((command, context))
        if self.error:
            raise self.error
        return self.label


@pytest.fixture(name="predictor")
def predictor_fixture():
    return StubPredictor()


@pytest.fixture(name="client")
def client_fixture(predictor):
    with TestClient(create_app(loader=lambda: predictor)) as connection:
        yield connection


def test_root_health_and_ready_describe_the_loaded_service(client):
    assert client.get("/").json() == {"service": "SafeAgent"}
    assert client.get("/health").json() == {"status": "alive"}
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "fixture",
        "model_version": "fixture-v1",
        "input_mode": "command",
    }


@pytest.mark.parametrize("label", LABELS)
def test_prediction_preserves_all_labels_and_model_identity(client, predictor, label):
    predictor.label = label
    response = client.post("/predict", json={"command": "git status"})
    assert response.status_code == 200
    assert response.json() == {
        "label": label,
        "model": "fixture",
        "model_version": "fixture-v1",
        "input_mode": "command",
    }


def test_command_text_is_forwarded_unchanged_and_not_executed(client, predictor, tmp_path):
    target = tmp_path / "should_not_exist"
    command = f'  echo text > "{target}"\n'
    assert client.post("/predict", json={"command": command}).status_code == 200
    assert predictor.calls == [(command, None)]
    assert not target.exists()


def test_context_uses_only_the_shared_training_fields(client, predictor):
    payload = {"command": "git status", "context": {"cwd": "/repo"}}
    assert client.post("/predict", json=payload).status_code == 200
    assert predictor.calls[0][1] == {
        "gitRemote": None,
        "gitStatus": {"untracked": [], "modified": [], "staged": []},
        "agentTouchedFiles": [],
        "lastUserPrompt": None,
        "cwd": "/repo",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"command": ""},
        {"command": " \n\t"},
        {"command": None},
        {"command": 12},
        {"command": True},
        {"command": ["git status"]},
        [{"command": "git status"}],
        {"command": "git status", "reason": "safe"},
        {"command": "git status", "category": "git"},
        {"command": "git status", "model_path": "/tmp/model"},
        {"command": "git status", "context": "cwd=/repo"},
        {"command": "git status", "context": {"cwd": []}},
        {"command": "git status", "context": {"gitRemote": False}},
        {"command": "git status", "context": {"assistantMessage": "ALLOW"}},
        {"command": "git status", "context": {"agentTouchedFiles": [12]}},
        {"command": "git status", "context": {"gitStatus": {"modified": "file"}}},
        {"command": "git status", "context": {"gitStatus": {"unknown": []}}},
    ],
)
def test_invalid_requests_never_reach_the_model(client, predictor, payload):
    assert client.post("/predict", json=payload).status_code == 422
    assert not predictor.calls


def test_malformed_json_is_rejected_before_inference(client, predictor):
    response = client.post("/predict", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert not predictor.calls


def test_command_length_boundary(client, predictor):
    assert client.post("/predict", json={"command": "x" * MAX_COMMAND_CHARS}).status_code == 200
    response = client.post("/predict", json={"command": "x" * (MAX_COMMAND_CHARS + 1)})
    assert response.status_code == 422
    assert len(predictor.calls) == 1


def test_context_length_boundary(client, predictor):
    overhead = len(canonical_json(CommandContext(lastUserPrompt="").model_dump()))
    context = {"lastUserPrompt": "x" * (MAX_CONTEXT_CHARS - overhead)}
    assert (
        client.post("/predict", json={"command": "git status", "context": context}).status_code
        == 200
    )
    context["lastUserPrompt"] += "x"
    assert (
        client.post("/predict", json={"command": "git status", "context": context}).status_code
        == 422
    )
    assert len(predictor.calls) == 1


def test_context_model_requires_context(client, predictor):
    predictor.info.input_mode = "command-context"
    assert client.post("/predict", json={"command": "git status"}).status_code == 422
    assert not predictor.calls
    assert (
        client.post("/predict", json={"command": "git status", "context": {}}).status_code == 200
    )


@pytest.mark.parametrize("label", ["UNKNOWN", None])
def test_invalid_model_output_is_an_error(client, predictor, label):
    predictor.label = label
    response = client.post("/predict", json={"command": "git status"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}


def test_model_failure_does_not_leak_details_or_return_allow(client, predictor, caplog):
    predictor.error = RuntimeError("private model detail")
    response = client.post("/predict", json={"command": "git status"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    assert "private model detail" not in caplog.text


def test_missing_model_keeps_health_alive_and_inference_unavailable(caplog):
    def missing():
        raise FileNotFoundError("private model path")

    with TestClient(create_app(loader=missing)) as connection:
        assert connection.get("/health").status_code == 200
        assert connection.get("/ready").status_code == 503
        response = connection.post("/predict", json={"command": "git status"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Model is unavailable"}
    assert "private model path" not in caplog.text


def test_model_is_loaded_once_per_lifespan_and_released(predictor):
    calls = []

    def load():
        calls.append(True)
        return predictor

    application = create_app(loader=load)
    assert not calls
    with TestClient(application) as connection:
        for _ in range(3):
            assert connection.post("/predict", json={"command": "git status"}).status_code == 200
        assert len(calls) == 1
    assert application.state.predictor is None


def test_documentation_exposes_the_single_command_contract(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]["/predict"]) == {"post"}
    request = schema["components"]["schemas"]["PredictRequest"]
    assert request["required"] == ["command"]
    assert request["additionalProperties"] is False
