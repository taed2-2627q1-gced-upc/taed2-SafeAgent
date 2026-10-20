"""Measure CPU inference energy and latency across baseline and candidate models.

Implements controlled batch benchmarking simulating a 1 vCPU runtime with CodeCarbon.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

from codecarbon import OfflineEmissionsTracker
import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import typer

from taed2_safeagent.config import PROJ_ROOT, REPORTS_DIR
from taed2_safeagent.data.common import load_params, write_json
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import load_examples

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_COUNTRY = "ESP"  # Location of Virtech (Spain)
DEFAULT_OUTPUT = REPORTS_DIR / "inference/energy.json"

# Reference CPU latency benchmarks from task D4 (used when weights are not cached locally)
D4_BENCHMARK_REFERENCE = {
    "codebert__command__2__a740d875": {
        "median_latency_ms": 831.00,
        "p95_latency_ms": 866.43,
        "mean_latency_ms": 835.20,
        "queries_count": 3,
        "duration_seconds": 2.5056,
        "model_size_mb": 481.50,
    },
    "codebert__command-context__2__a740d875": {
        "median_latency_ms": 12639.40,
        "p95_latency_ms": 12801.71,
        "mean_latency_ms": 12650.10,
        "queries_count": 3,
        "duration_seconds": 37.9503,
        "model_size_mb": 481.50,
    },
    "modernbert__command__2__a740d875": {
        "median_latency_ms": 5390.19,
        "p95_latency_ms": 6105.20,
        "mean_latency_ms": 5420.30,
        "queries_count": 3,
        "duration_seconds": 16.2609,
        "model_size_mb": 575.55,
    },
}


def benchmark_svm(pipeline: Any, texts: list[str], warmup: int = 5) -> list[float]:
    """Run sequential inference on SVM pipeline and collect latency measurements."""
    for i in range(min(warmup, len(texts))):
        pipeline.predict([texts[i]])

    latencies: list[float] = []
    for text in texts:
        t0 = perf_counter()
        pipeline.predict([text])
        latencies.append((perf_counter() - t0) * 1000.0)
    return latencies


def benchmark_encoder(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    warmup: int = 1,
    max_length: int = 512,
) -> list[float]:
    """Run single-query sequential inference on PyTorch encoder model."""
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    for i in range(min(warmup, len(texts))):
        inputs = tokenizer(
            [texts[i]],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            model(**inputs)

    latencies: list[float] = []
    for text in texts:
        inputs = tokenizer(
            [text],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        ).to(device)
        t0 = perf_counter()
        with torch.inference_mode():
            model(**inputs)
        latencies.append((perf_counter() - t0) * 1000.0)
    return latencies


def measure_inference_batch(
    runner_fn: Callable[[], list[float]],
    queries_count: int,
    country_iso_code: str = DEFAULT_COUNTRY,
) -> dict[str, Any]:
    """Track process energy consumption during execution of the inference runner."""
    tracker = None
    try:
        tracker = OfflineEmissionsTracker(
            country_iso_code=country_iso_code,
            tracking_mode="process",
            measure_power_secs=1,
            save_to_file=False,
            log_level="warning",
        )
        tracker.start()
    except (ValueError, OSError, RuntimeError):
        tracker = None

    t_start = perf_counter()
    latencies = runner_fn()
    duration_sec = perf_counter() - t_start

    energy_kwh = 0.0
    emissions_kg = 0.0

    if tracker is not None:
        try:
            tracker.stop()
            emissions_data = tracker.final_emissions_data
            if emissions_data is not None:
                energy_kwh = float(emissions_data.energy_consumed or 0.0)
                emissions_kg = float(emissions_data.emissions or 0.0)
        except (ValueError, OSError, RuntimeError):
            pass

    energy_wh = energy_kwh * 1000.0
    energy_per_query_wh = (energy_wh / queries_count) if queries_count > 0 else 0.0
    emissions_per_query_mg = ((emissions_kg * 1e6) / queries_count) if queries_count > 0 else 0.0

    return {
        "status": "measured",
        "queries_count": queries_count,
        "duration_seconds": round(duration_sec, 4),
        "energy_consumed_kwh": energy_kwh,
        "emissions_kg": emissions_kg,
        "energy_per_query_wh": round(energy_per_query_wh, 6),
        "emissions_per_query_mg": round(emissions_per_query_mg, 6),
        "median_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
    }


def is_model_ready(model_path: Path, model_type: str) -> bool:
    """Check if model weights and tokenizer configuration exist locally."""
    if model_type == "svm":
        return model_path.exists()
    return (model_path / "config.json").exists() and (
        (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists()
    )


def estimate_from_d4_benchmark(
    model_name: str,
    country_iso_code: str,
    cpu_power_w: float = 64.0,
) -> dict[str, Any] | None:
    """Calculate energy estimate from D4 benchmark latency and CPU TDP."""
    ref = D4_BENCHMARK_REFERENCE.get(model_name)
    if ref is None:
        return None

    duration_sec = ref["duration_seconds"]
    queries_count = ref["queries_count"]
    energy_kwh = (cpu_power_w * duration_sec) / 3_600_000.0

    # Emission factor (approx 0.15 kg CO2/kWh for Spain grid or 0.38 for US)
    factor = 0.15 if country_iso_code == "ESP" else 0.38
    emissions_kg = energy_kwh * factor

    energy_wh = energy_kwh * 1000.0
    energy_per_query_wh = (energy_wh / queries_count) if queries_count > 0 else 0.0
    emissions_per_query_mg = ((emissions_kg * 1e6) / queries_count) if queries_count > 0 else 0.0

    return {
        "status": "estimated_from_d4_latency",
        "queries_count": queries_count,
        "duration_seconds": round(duration_sec, 4),
        "energy_consumed_kwh": energy_kwh,
        "emissions_kg": emissions_kg,
        "energy_per_query_wh": round(energy_per_query_wh, 6),
        "emissions_per_query_mg": round(emissions_per_query_mg, 6),
        "median_latency_ms": ref["median_latency_ms"],
        "p95_latency_ms": ref["p95_latency_ms"],
        "mean_latency_ms": ref["mean_latency_ms"],
    }


def evaluate_single_model(
    cfg: dict[str, Any],
    val_rows: list[dict[str, Any]],
    country_iso_code: str,
) -> dict[str, Any] | None:
    """Evaluate energy and latency for one model configuration."""
    model_name = cfg["name"]
    model_path = cfg["path"]
    input_mode = cfg["input_mode"]
    num_q = cfg["queries"]

    if not is_model_ready(model_path, cfg["type"]):
        metrics = estimate_from_d4_benchmark(model_name, country_iso_code)
        if metrics is None:
            return None
        metrics["input_type"] = input_mode
        return metrics

    selected_rows = val_rows[:num_q]
    texts = [build_input(row, input_mode) for row in selected_rows]

    if cfg["type"] == "svm":
        pipeline = joblib.load(model_path)
        metrics = measure_inference_batch(
            lambda: benchmark_svm(pipeline, texts),
            num_q,
            country_iso_code=country_iso_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        metrics = measure_inference_batch(
            lambda: benchmark_encoder(model, tokenizer, texts),
            num_q,
            country_iso_code=country_iso_code,
        )

    metrics["input_type"] = input_mode

    report_dir = cfg["report_dir"]
    if report_dir.exists():
        indiv_report = {
            "model": model_name,
            "scope": "CPU inference on controlled validation batch",
            "hardware": {"device": "CPU", "threads": 1},
            "country_iso_code": country_iso_code,
            **metrics,
        }
        write_json(report_dir / "inference_energy.json", indiv_report)

    return metrics


def evaluate_all_models(
    val_rows: list[dict[str, Any]],
    svm_queries: int = 500,
    encoder_queries: int = 3,
    country_iso_code: str = DEFAULT_COUNTRY,
) -> dict[str, Any]:
    """Run energy and latency benchmark across the baseline and candidate encoders."""
    torch.set_num_threads(1)

    models_config = [
        {
            "name": "tfidf_char_linear_svm",
            "type": "svm",
            "path": PROJ_ROOT / "models/tfidf_char_linear_svm/pipeline.joblib",
            "report_dir": REPORTS_DIR / "baseline",
            "input_mode": "command",
            "queries": min(svm_queries, len(val_rows)),
        },
        {
            "name": "codebert__command__2__a740d875",
            "type": "encoder",
            "path": PROJ_ROOT / "models/encoders/codebert__command__2__a740d875",
            "report_dir": REPORTS_DIR / "encoders/codebert__command__2__a740d875",
            "input_mode": "command",
            "queries": min(encoder_queries, len(val_rows)),
        },
        {
            "name": "codebert__command-context__2__a740d875",
            "type": "encoder",
            "path": PROJ_ROOT / "models/encoders/codebert__command-context__2__a740d875",
            "report_dir": REPORTS_DIR / "encoders/codebert__command-context__2__a740d875",
            "input_mode": "command-context",
            "queries": min(encoder_queries, len(val_rows)),
        },
        {
            "name": "modernbert__command__2__a740d875",
            "type": "encoder",
            "path": PROJ_ROOT / "models/encoders/modernbert__command__2__a740d875",
            "report_dir": REPORTS_DIR / "encoders/modernbert__command__2__a740d875",
            "input_mode": "command",
            "queries": min(encoder_queries, len(val_rows)),
        },
        {
            "name": "modernbert__command-context__2__a740d875",
            "type": "encoder",
            "path": PROJ_ROOT / "models/encoders/modernbert__command-context__2__a740d875",
            "report_dir": REPORTS_DIR / "encoders/modernbert__command-context__2__a740d875",
            "input_mode": "command-context",
            "queries": min(encoder_queries, len(val_rows)),
        },
    ]

    results: dict[str, Any] = {}
    for cfg in models_config:
        typer.echo(f"Benchmarking model: {cfg['name']} ({cfg['input_mode']})...")
        metrics = evaluate_single_model(cfg, val_rows, country_iso_code)
        if metrics is not None:
            results[cfg["name"]] = metrics

    return {
        "scope": "CPU inference on controlled validation batch",
        "hardware": {"device": "CPU", "threads": 1},
        "country_iso_code": country_iso_code,
        "models": results,
    }


@app.command()
def main(
    svm_queries: Annotated[int, typer.Option(help="Number of queries for SVM baseline")] = 500,
    encoder_queries: Annotated[
        int, typer.Option(help="Number of queries for candidate encoders")
    ] = 3,
    output: Annotated[
        Path, typer.Option(help="Path to save consolidated report")
    ] = DEFAULT_OUTPUT,
) -> None:
    """Measure inference energy consumption across models and save reports."""
    params = load_params(PROJ_ROOT / "params.yaml")
    val_rows = load_examples(params, "validation")
    country = params.get("energy", {}).get("country_iso_code", DEFAULT_COUNTRY)

    typer.echo(f"Running inference energy benchmark (Country: {country})...")
    report = evaluate_all_models(
        val_rows,
        svm_queries=svm_queries,
        encoder_queries=encoder_queries,
        country_iso_code=country,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    typer.echo(f"Consolidated inference energy report saved to: {output}")


if __name__ == "__main__":
    app()
