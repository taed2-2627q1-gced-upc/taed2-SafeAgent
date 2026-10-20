import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from taed2_safeagent.config import MODEL_NAMES
from taed2_safeagent.data.inputs import build_input

# Configuración global del benchmark
REPETITION_COUNT = 20
ROWS_PER_REPETITION = 30


def main():
    torch.set_num_threads(os.cpu_count() or 4)

    cpu_metrics = {}

    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root: {project_root}")

    sample_data_path = project_root / "data/processed/shell_safety/test.jsonl"
    df = pd.read_json(sample_data_path, lines=True)

    sample_rows = df.sample(n=ROWS_PER_REPETITION, random_state=42).to_dict(
        orient="records"
    )

    sample_inputs = {
        "command": [build_input(row, "command") for row in sample_rows],
        "command-context": [
            build_input(row, "command-context") for row in sample_rows
        ],
    }

    # ---------------------------------------------------------
    # 1. Evaluación de Encoders (Transformers)
    # ---------------------------------------------------------
    for model_name in MODEL_NAMES:
        print("\nEvaluating model:", model_name)
        model_dir = project_root / "models/encoders" / model_name

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        input_type = (
            "command-context" if "context" in model_name else "command"
        )
        inputs = tokenizer(
            sample_inputs[input_type],
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to("cpu")

        start = time.perf_counter()
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        ).to("cpu")
        load_time = time.perf_counter() - start
        model.eval()

        model_size_mb = sum(
            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)

        # Warm-up
        with torch.no_grad():
            _ = model(**inputs)

        latencies_list = []
        for i in range(REPETITION_COUNT):
            print(
                f" - Running inference for {model_name}, iteration {i + 1}",
                end="",
            )
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            t1 = time.perf_counter()

            latencies_list.append((t1 - t0) * 1000.0)
            print(f" - Latency: {latencies_list[-1]:.2f} ms")

        cpu_metrics[model_name] = {
            "input_type": input_type,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "median_latency_ms": float(np.median(latencies_list)),
            "p95_latency_ms": float(np.percentile(latencies_list, 95)),
            "repetition_count": REPETITION_COUNT,
        }

    # ---------------------------------------------------------
    # 2. Evaluación del Baseline (Scikit-Learn)
    # ---------------------------------------------------------
    model_dir = project_root / "models/tfidf_char_linear_svm"
    model_name = "tfidf_char_linear_svm"

    print("\nEvaluating model:", model_name)

    start = time.perf_counter()
    pipeline = joblib.load(model_dir / "pipeline.joblib")
    load_time = time.perf_counter() - start

    model_size_mb = sum(
        f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    # Warm-up
    _ = pipeline.predict(sample_inputs["command"])

    latencies_list = []
    for i in range(REPETITION_COUNT):
        print(
            f" - Running inference for {model_name}, iteration {i + 1}",
            end="",
        )
        t0 = time.perf_counter()
        _ = pipeline.predict(sample_inputs["command"])
        t1 = time.perf_counter()

        latencies_list.append((t1 - t0) * 1000.0)
        print(f" - Latency: {latencies_list[-1]:.2f} ms")

    cpu_metrics[model_name] = {
        "input_type": "command",
        "model_size_mb": model_size_mb,
        "load_time_seconds": load_time,
        "median_latency_ms": float(np.median(latencies_list)),
        "p95_latency_ms": float(np.percentile(latencies_list, 95)),
        "repetition_count": REPETITION_COUNT,
    }

    # ---------------------------------------------------------
    # 3. Exportación de métricas
    # ---------------------------------------------------------
    output_path = Path(__file__).parent / "cpu_metrics.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cpu_metrics, f, indent=4)

    print(f"\nBenchmark completado. Métricas guardadas en: {output_path}")


if __name__ == "__main__":
    main()