"""Check that the demo detects service and model identity failures."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from scripts import smoke_test

from taed2_safeagent.api.app import create_app
from taed2_safeagent.api.schemas import ModelInfo

VERSION = "sha256:" + "a" * 64


@pytest.fixture(name="demo_service")
def demo_service_fixture(monkeypatch):
    predictor = SimpleNamespace(
        info=ModelInfo(model="fixture", model_version=VERSION, input_mode="command"),
        predict=lambda command, _context: "DENY" if command == "rm -rf /" else "ALLOW",
    )
    with TestClient(create_app(loader=lambda: predictor)) as client:

        def send_request(_url, method, path, payload=None):
            response = client.request(method, path, json=payload)
            if response.headers.get("content-type", "").startswith("application/json"):
                body = response.json()
            else:
                body = response.text
            return response.status_code, body, 0.0

        monkeypatch.setattr(smoke_test, "send_request", send_request)
        yield send_request


def test_demo_checks_documentation_prediction_and_version(demo_service, capsys):
    assert demo_service
    assert smoke_test.run_demo("http://service", VERSION)
    assert "7/7 checks passed" in capsys.readouterr().out


def test_demo_rejects_an_unexpected_model_version(demo_service, capsys):
    assert demo_service
    assert not smoke_test.run_demo("http://service", "sha256:" + "b" * 64)
    assert "differs from the expected version" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["unavailable", "changed_identity", "broken_schema"])
def test_demo_fails_for_service_and_contract_errors(demo_service, monkeypatch, fault):
    def send_request(url, method, path, payload=None):
        if fault == "unavailable" and path == "/ready":
            return 503, {"detail": "Model is unavailable"}, 0.0
        status, body, latency = demo_service(url, method, path, payload)
        if fault == "changed_identity" and path == "/predict" and status == 200:
            body["model_version"] = "another model"
        if fault == "broken_schema" and path == "/openapi.json":
            body["paths"]["/predict"]["post"]["responses"].pop("503")
        return status, body, latency

    monkeypatch.setattr(smoke_test, "send_request", send_request)
    assert not smoke_test.run_demo("http://service", VERSION)
