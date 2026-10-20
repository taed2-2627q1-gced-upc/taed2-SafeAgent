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


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test SafeAgent API on Virtech")
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    success = run_smoke_tests(args.url)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

