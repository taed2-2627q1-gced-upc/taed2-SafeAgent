import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from taed2_safeagent.config import MODEL_NAMES
from taed2_safeagent.data.inputs import build_input

torch.set_num_threads(os.cpu_count() or 4)


repetition_count = 15
rows_per_repetition = 30
cpu_metrics = {}


project_root = Path(__file__).resolve().parents[2]
print(f"Project root: {project_root}")

sample_data_path = project_root / "data/processed/shell_safety/test.jsonl"
df = pd.read_json(sample_data_path, lines=True)

sample_rows = df.sample(n=rows_per_repetition, random_state=42).to_dict(orient="records")

sample_inputs = {
    "command": [build_input(row, "command") for row in sample_rows],
    "command-context": [build_input(row, "command-context") for row in sample_rows],
}


for model_name in MODEL_NAMES:
    print("\nEvaluating model:", model_name)
    model_dir = project_root / "models/encoders/" / model_name

    # Cargar tokenizer y tokenizar la lista de textos
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    input_type = "command-context" if "context" in model_name else "command"
    inputs = tokenizer(
        sample_inputs[input_type],
        padding="longest",  # Antes era True pero tardaba mucho
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cpu")

    # Cargar modelo y ver cuánto se tarda en ello
    start = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to("cpu")
    load_time = time.perf_counter() - start
    model.eval()

    # Medir cuánto ocupa
    model_size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (
        1024 * 1024
    )

    # Warm-up
    with torch.no_grad():
        _ = model(**inputs)

    # Evaluar con todos los inputs durante repetition_count repeticiones
    latencies_list = []
    for i in range(repetition_count):
        print(f" - Running inference for {model_name}, iteration {i + 1}", end="")
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        t1 = time.perf_counter()

        # Guardar y convertir a ms
        latencies_list.append((t1 - t0) * 1000.0)
        print(f" - Latency: {latencies_list[-1]:.2f} ms")

    cpu_metrics[model_name] = {
        "input_type": input_type,
        "model_size_mb": model_size_mb,
        "load_time_seconds": load_time,
        "median_latency_ms": float(np.median(latencies_list)),
        "p95_latency_ms": float(np.percentile(latencies_list, 95)),
        "repetition_count": repetition_count,
    }


"""Inferencia para el modelo baseline"""

import joblib

model_dir = project_root / "models/tfidf_char_linear_svm"
model_name = "tfidf_char_linear_svm"

print("Evaluating model:", model_name)

# Cargar modelo y ver cuánto se tarda en ello
start = time.perf_counter()
pipeline = joblib.load(model_dir / "pipeline.joblib")
load_time = time.perf_counter() - start

# Medir cuánto ocupa
model_size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024 * 1024)


# Evaluar con todos los inputs durante repetition_count repeticiones
latencies_list = []
for i in range(repetition_count):
    print(f" - Running inference for {model_name}, iteration {i + 1}", end="")
    t0 = time.perf_counter()
    predicted_labels = pipeline.predict(sample_inputs["command"])
    t1 = time.perf_counter()

    # Guardar y convertir a ms
    latencies_list.append((t1 - t0) * 1000.0)
    print(f" - Latency: {latencies_list[-1]:.2f} ms")


cpu_metrics[model_name] = {
    "input_type": "command",
    "model_size_mb": model_size_mb,
    "load_time_seconds": load_time,
    "median_latency_ms": float(np.median(latencies_list)),
    "p95_latency_ms": float(np.percentile(latencies_list, 95)),
    "repetition_count": repetition_count,
}


output_path = Path(__file__).parent / "cpu_metrics.json"

# Guardar el diccionario con formato legible (indent=4)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cpu_metrics, f, indent=4)
