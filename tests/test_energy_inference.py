from pathlib import Path
from unittest.mock import MagicMock

import pytest

from taed2_safeagent.modeling.energy_inference import (
    benchmark_svm,
    estimate_from_d4_benchmark,
    evaluate_single_model,
    measure_inference_batch,
)


def test_benchmark_svm_computes_latencies():
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = ["ALLOW"]
    texts = ["git status", "ls -la", "pwd"]

    latencies = benchmark_svm(mock_pipeline, texts, warmup=1)

    assert len(latencies) == len(texts)
    assert all(lat >= 0.0 for lat in latencies)
    assert mock_pipeline.predict.call_count == len(texts) + 1  # warmup + texts


def test_measure_inference_batch_returns_expected_structure():
    dummy_latencies = [12.0, 14.5, 18.2, 20.0]

    def runner_fn():
        return dummy_latencies

    metrics = measure_inference_batch(runner_fn, queries_count=len(dummy_latencies))

    required_keys = {
        "queries_count",
        "duration_seconds",
        "energy_consumed_kwh",
        "emissions_kg",
        "energy_per_query_wh",
        "emissions_per_query_mg",
        "median_latency_ms",
        "p95_latency_ms",
        "mean_latency_ms",
    }
    assert required_keys.issubset(metrics.keys())
    assert metrics["queries_count"] == 4
    assert metrics["median_latency_ms"] == pytest.approx(16.35, abs=0.5)


def test_evaluate_single_model_skips_missing_path(tmp_path: Path):
    cfg = {
        "name": "non_existent_model",
        "type": "svm",
        "path": tmp_path / "does_not_exist.joblib",
        "report_dir": tmp_path / "reports",
        "input_mode": "command",
        "queries": 5,
    }
    result = evaluate_single_model(cfg, val_rows=[], country_iso_code="ESP")
    assert result is None


def test_estimate_from_d4_benchmark():
    metrics = estimate_from_d4_benchmark("codebert__command__2__a740d875", "ESP")
    assert metrics is not None
    assert metrics["queries_count"] == 3
    assert metrics["median_latency_ms"] == 831.00
    assert metrics["energy_consumed_kwh"] > 0
