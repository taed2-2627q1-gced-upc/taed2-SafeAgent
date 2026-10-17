import hashlib
import json
import socket
from threading import Thread
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import uvicorn

from taed2_safeagent.api import smoke
from taed2_safeagent.api.app import create_app
from taed2_safeagent.api.model import DEFAULT_MODEL_DIR
from taed2_safeagent.api.schemas import ModelInfo


@pytest.fixture(name="serve")
def local_http_server():
    servers = []

    def start(application):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(application, log_level="error", lifespan="on"))
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        servers.append((server, thread, listener))
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("Local HTTP server did not start")
            time.sleep(0.01)
        return f"http://127.0.0.1:{listener.getsockname()[1]}"

    yield start
    for server, thread, listener in servers:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()


@pytest.fixture(name="predictor")
def saved_predictor():
    return SimpleNamespace(
        info=ModelInfo(model="fixture", model_version="fixture-v1", input_mode="command"),
        predict=lambda command, context: "ASK",
    )


def test_smoke_check_uses_real_http_and_preserves_model_identity(serve, predictor):
    url = serve(create_app(loader=lambda: predictor))
    result = smoke.check_service(url + "/", expected_version="fixture-v1")
    assert result == {
        "status": "passed",
        "model": "fixture",
        "model_version": "fixture-v1",
        "input_mode": "command",
        "label": "ASK",
        "checks": ["health", "ready", "prediction", "empty_command", "unknown_fields", "batch"],
    }


def test_smoke_cli_reports_success_for_a_context_predictor(serve, predictor, capsys):
    predictor.info.input_mode = "command-context"
    url = serve(create_app(loader=lambda: predictor))
    assert smoke.main(["--url", url, "--expect-model-version", "fixture-v1"]) == 0
    assert json.loads(capsys.readouterr().out)["input_mode"] == "command-context"


def test_unavailable_model_is_a_failed_smoke_check(serve, capsys):
    def missing():
        raise FileNotFoundError("private path")

    url = serve(create_app(loader=missing))
    assert smoke.main(["--url", url]) == 1
    error = capsys.readouterr().err
    assert "HTTP 200 from /ready, received 503" in error
    assert "private path" not in error


def test_wrong_model_version_fails(serve, predictor):
    url = serve(create_app(loader=lambda: predictor))
    with pytest.raises(smoke.SmokeCheckError, match="expected version"):
        smoke.check_service(url, expected_version="other-model")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_cli_rejects_invalid_timeout_without_network(timeout, capsys):
    assert smoke.main(["--url", "http://127.0.0.1:8000", "--timeout", str(timeout)]) == 1
    assert "positive finite" in capsys.readouterr().err


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model",
        "http:///health",
        "http://user:secret@host",
        "https://host?token=secret",
        "https://host#fragment",
    ],
)
def test_cli_rejects_invalid_service_url_without_network(url, capsys):
    assert smoke.main(["--url", url]) == 1
    error = capsys.readouterr().err
    assert "service URL" in error
    assert "secret" not in error


def test_connection_failure_returns_a_nonzero_exit(capsys):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{reservation.getsockname()[1]}"
        assert smoke.main(["--url", url, "--timeout", "0.1"]) == 1
    assert "Could not reach /health" in capsys.readouterr().err


@pytest.mark.parametrize(
    "path,body,message",
    [
        ("/health", {"status": "ready"}, "health"),
        ("/ready", {"status": "alive"}, "readiness"),
        ("/ready", {"status": "ready", "model": "fixture"}, "model identity"),
        ("/predict", {"label": "UNKNOWN"}, "prediction response"),
        (
            "/predict",
            {
                "label": "ASK",
                "model": "fixture",
                "model_version": "changed",
                "input_mode": "command",
            },
            "differs from readiness",
        ),
    ],
)
def test_incorrect_http_contract_fails(monkeypatch, path, body, message):
    responses = {
        "/health": {"status": "alive"},
        "/ready": {
            "status": "ready",
            "model": "fixture",
            "model_version": "fixture-v1",
            "input_mode": "command",
        },
    }
    responses[path] = body
    monkeypatch.setattr(
        smoke, "_request", lambda base, route, timeout, payload=None: (200, responses[route])
    )
    with pytest.raises(smoke.SmokeCheckError, match=message):
        smoke.check_service("http://127.0.0.1:8000")


@pytest.mark.parametrize("content", [b"<html>Proxy error</html>", b"\xff"])
def test_non_json_http_response_fails(monkeypatch, content):
    connection = MagicMock()
    connection.status = 200
    connection.read.return_value = content
    monkeypatch.setattr(smoke, "urlopen", lambda *args, **kwargs: connection)
    with pytest.raises(smoke.SmokeCheckError, match="Expected JSON"):
        smoke.check_service("http://127.0.0.1:8000")


def test_response_timeout_fails_without_exposing_details(monkeypatch, capsys):
    connection = MagicMock()
    connection.read.side_effect = TimeoutError("private connection detail")
    monkeypatch.setattr(smoke, "urlopen", lambda *args, **kwargs: connection)
    assert smoke.main(["--url", "http://127.0.0.1:8000"]) == 1
    error = capsys.readouterr().err
    assert "Could not read /health" in error
    assert "private connection detail" not in error


@pytest.mark.integration
def test_saved_baseline_passes_http_smoke_check(serve, monkeypatch):
    artifact = DEFAULT_MODEL_DIR / "pipeline.joblib"
    if not artifact.is_file():
        pytest.skip("Recover models/tfidf_char_linear_svm with DVC first")
    monkeypatch.setenv("SAFEAGENT_MODEL_DIR", str(DEFAULT_MODEL_DIR))
    version = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    url = serve(create_app())
    assert smoke.check_service(url, expected_version=version)["status"] == "passed"
