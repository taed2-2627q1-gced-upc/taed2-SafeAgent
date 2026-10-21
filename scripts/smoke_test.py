#!/usr/bin/env python3
"""Smoke test client for SafeAgent API.

Validates that the deployed service on Virtech handles valid and invalid requests
as specified in the HTTP contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://nattech.fib.upc.edu:40490"
IDENTITY_FIELDS = ("model", "model_version", "input_mode")


def send_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | str | None = None,
    timeout: float = 10.0,
) -> tuple[int, Any, float]:
    """Send an HTTP request and measure response time."""
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    data = None
    headers = {}

    if payload is not None:
        headers["Content-Type"] = "application/json"
        if isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    start_time = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            body = response.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = body
            return response.status, parsed, latency_ms
    except urllib.error.HTTPError as error:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        body = error.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return error.code, parsed, latency_ms
    except (urllib.error.URLError, OSError) as error:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return 0, str(error), latency_ms


def run_smoke_tests(base_url: str) -> bool:
    """Execute all smoke test cases against the API."""
    print("=== SafeAgent API Smoke Tests ===")
    print(f"Target URL: {base_url}\n")

    test_cases = [
        # (name, method, path, payload, expected_status, validator_fn)
        (
            "Service identification (GET /)",
            "GET",
            "/",
            None,
            200,
            lambda res: res.get("service") == "SafeAgent",
        ),
        (
            "Liveness probe (GET /health)",
            "GET",
            "/health",
            None,
            200,
            lambda res: res.get("status") == "alive",
        ),
        (
            "Readiness probe (GET /ready)",
            "GET",
            "/ready",
            None,
            200,
            lambda res: res.get("status") == "ready" and "model" in res,
        ),
        (
            "Valid command prediction: benign (POST /predict)",
            "POST",
            "/predict",
            {"command": "git status"},
            200,
            lambda res: res.get("label") in ("ALLOW", "ASK", "DENY") and "model" in res,
        ),
        (
            "Valid command prediction: dangerous (POST /predict)",
            "POST",
            "/predict",
            {"command": "rm -rf /"},
            200,
            lambda res: res.get("label") == "DENY",
        ),
        (
            "Valid prediction with context (POST /predict)",
            "POST",
            "/predict",
            {
                "command": "cat README.md",
                "context": {"cwd": "/home/user/project", "lastUserPrompt": "read readme"},
            },
            200,
            lambda res: res.get("label") in ("ALLOW", "ASK", "DENY"),
        ),
        (
            "Invalid command: empty string (POST /predict)",
            "POST",
            "/predict",
            {"command": ""},
            422,
            lambda res: "detail" in res,
        ),
        (
            "Invalid command: whitespace only (POST /predict)",
            "POST",
            "/predict",
            {"command": "   "},
            422,
            lambda res: "detail" in res,
        ),
        (
            "Invalid command: wrong type integer (POST /predict)",
            "POST",
            "/predict",
            {"command": 12345},
            422,
            lambda res: "detail" in res,
        ),
        (
            "Invalid payload: unknown extra field (POST /predict)",
            "POST",
            "/predict",
            {"command": "ls -la", "extra_unsupported_field": "value"},
            422,
            lambda res: "detail" in res,
        ),
        (
            "Non-existent route (GET /nonexistent)",
            "GET",
            "/nonexistent",
            None,
            404,
            lambda res: True,
        ),
    ]

    all_passed = True
    passed_count = 0

    for name, method, path, payload, expected_status, validator in test_cases:
        status, response, latency = send_request(base_url, method, path, payload)
        passed_status = status == expected_status
        passed_validation = passed_status and validator(response)

        if passed_validation:
            passed_count += 1
            status_indicator = "PASS"
        else:
            all_passed = False
            status_indicator = "FAIL"

        print(
            f"[{status_indicator}] {name:<55} "
            f"HTTP {status} (expected {expected_status}) "
            f"{latency:6.1f}ms"
        )
        if not passed_validation:
            print(f"       Details: {response}")

    print("\n" + "=" * 50)
    print(f"Result: {passed_count}/{len(test_cases)} tests passed.")

    return all_passed


def run_demo(base_url: str, expected_version: str | None = None) -> bool:
    """Check the public service and show a short presentation demo."""
    print(f"SafeAgent live demo: {base_url}\n")
    status, ready, _ = send_request(base_url, "GET", "/ready")
    if (
        status != 200
        or not isinstance(ready, dict)
        or ready.get("status") != "ready"
        or any(not isinstance(ready.get(key), str) or not ready[key] for key in IDENTITY_FIELDS)
    ):
        print(f"[FAIL] Model readiness: HTTP {status}")
        return False
    identity = {key: ready[key] for key in IDENTITY_FIELDS}
    if expected_version is not None and identity["model_version"] != expected_version:
        print("[FAIL] The loaded model differs from the expected version")
        return False
    print("[PASS] Model ready")
    print(json.dumps(identity, indent=2))

    def valid_prediction(response, label):
        return (
            isinstance(response, dict)
            and response.get("label") == label
            and all(response.get(key) == value for key, value in identity.items())
        )

    cases = [
        (
            "Application alive",
            "GET",
            "/health",
            None,
            200,
            lambda body: body == {"status": "alive"},
        ),
        (
            "Swagger documentation",
            "GET",
            "/docs",
            None,
            200,
            lambda body: isinstance(body, str) and "SwaggerUIBundle" in body,
        ),
        (
            "OpenAPI schema and documented errors",
            "GET",
            "/openapi.json",
            None,
            200,
            lambda body: (
                isinstance(body, dict)
                and {"200", "422", "503", "500"}.issubset(
                    body.get("paths", {}).get("/predict", {}).get("post", {}).get("responses", {})
                )
            ),
        ),
        (
            "git status returns ALLOW",
            "POST",
            "/predict",
            {"command": "git status"},
            200,
            lambda body: valid_prediction(body, "ALLOW"),
        ),
        (
            "Destructive command returns DENY",
            "POST",
            "/predict",
            {"command": "rm -rf /"},
            200,
            lambda body: valid_prediction(body, "DENY"),
        ),
        (
            "Empty command is rejected",
            "POST",
            "/predict",
            {"command": ""},
            422,
            lambda body: isinstance(body, dict) and "detail" in body,
        ),
    ]
    passed = 1
    for name, method, path, payload, expected_status, validator in cases:
        status, response, _ = send_request(base_url, method, path, payload)
        success = status == expected_status and validator(response)
        passed += int(success)
        print(f"[{'PASS' if success else 'FAIL'}] {name}: HTTP {status}")
    print(f"\nResult: {passed}/{len(cases) + 1} checks passed")
    print("Commands were submitted as text and were never executed by this client.")
    return passed == len(cases) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test SafeAgent API on Virtech")
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument("--demo", action="store_true", help="Run the short Swagger and model demo")
    parser.add_argument("--expect-model-version", help="Required model version for the short demo")
    args = parser.parse_args()
    if args.expect_model_version and not args.demo:
        parser.error("--expect-model-version requires --demo")
    success = (
        run_demo(args.url, args.expect_model_version) if args.demo else run_smoke_tests(args.url)
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
