import argparse
from http.client import HTTPException
import json
import math
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import ValidationError

from taed2_safeagent.api.schemas import ModelInfo, PredictResponse


class SmokeCheckError(ValueError):
    """Describe a failed service check without response contents."""


def _request(base_url, path, timeout, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        response = error
    except (URLError, OSError, HTTPException) as error:
        raise SmokeCheckError(f"Could not reach {path}") from error
    with response:
        try:
            body = json.loads(response.read())
        except (OSError, HTTPException) as error:
            raise SmokeCheckError(f"Could not read {path}") from error
        except ValueError as error:
            raise SmokeCheckError(f"Expected JSON from {path}") from error
        return response.status, body


def _expect_status(response, expected, path):
    status, body = response
    if status != expected:
        raise SmokeCheckError(f"Expected HTTP {expected} from {path}, received {status}")
    return body


def check_service(base_url, timeout=5.0, expected_version=None):
    """Check a running API through HTTP without local model files."""
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SmokeCheckError("Use an HTTP or HTTPS service URL without credentials or query")
    if not math.isfinite(timeout) or timeout <= 0:
        raise SmokeCheckError("Timeout must be a positive finite number")
    base_url = base_url.rstrip("/")
    health = _expect_status(_request(base_url, "/health", timeout), 200, "/health")
    if health != {"status": "alive"}:
        raise SmokeCheckError("Unexpected health response")
    ready = _expect_status(_request(base_url, "/ready", timeout), 200, "/ready")
    if not isinstance(ready, dict) or ready.get("status") != "ready":
        raise SmokeCheckError("Unexpected readiness response")
    try:
        identity = ModelInfo.model_validate(
            {key: value for key, value in ready.items() if key != "status"}
        )
    except ValidationError as error:
        raise SmokeCheckError("Invalid model identity from /ready") from error
    if expected_version is not None and identity.model_version != expected_version:
        raise SmokeCheckError("Loaded model version differs from the expected version")
    payload = {"command": "git status", "context": {}}
    result = _expect_status(_request(base_url, "/predict", timeout, payload), 200, "/predict")
    try:
        prediction = PredictResponse.model_validate(result)
    except ValidationError as error:
        raise SmokeCheckError("Invalid prediction response") from error
    if prediction.model_dump(exclude={"label"}) != identity.model_dump():
        raise SmokeCheckError("Prediction model identity differs from readiness")
    invalid_payloads = ({"command": ""}, {"command": "git status", "reason": "safe"}, [payload])
    for invalid in invalid_payloads:
        _expect_status(_request(base_url, "/predict", timeout, invalid), 422, "/predict")
    return {
        "status": "passed",
        **identity.model_dump(),
        "label": prediction.label,
        "checks": ["health", "ready", "prediction", "empty_command", "unknown_fields", "batch"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check a running SafeAgent API through HTTP")
    parser.add_argument("--url", required=True, help="Service base URL")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds per HTTP request")
    parser.add_argument(
        "--expect-model-version", help="Required model version from a known artifact"
    )
    options = parser.parse_args(argv)
    try:
        result = check_service(options.url, options.timeout, options.expect_model_version)
    except ValueError as error:
        print(f"Smoke check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
