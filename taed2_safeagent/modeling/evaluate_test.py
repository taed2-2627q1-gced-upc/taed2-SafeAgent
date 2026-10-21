# pylint: disable=duplicate-code
import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from taed2_safeagent.config import MODEL_NAMES, PROJ_ROOT
from taed2_safeagent.data.inputs import build_input


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula las métricas de evaluación para las 3 clases (ALLOW, DENY, ASK)."""
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    count = len(df)
    critical_cases = int(((y_true == "DENY") & (y_pred == "ALLOW")).sum())
    accuracy = float((y_true == y_pred).mean())

    labels = ["DENY", "ASK", "ALLOW"]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    metrics = {
        "count": count,
        "accuracy": accuracy,
        "critical_cases": critical_cases,
        "critical_cases_ratio": float(critical_cases / count) if count > 0 else 0.0,
    }

    # Agregar F1, Precision y Recall para cada una de las 3 clases
    for idx, label in enumerate(labels):
        lbl_lower = label.lower()
        metrics[f"f1_{lbl_lower}"] = float(f1[idx])
        metrics[f"precision_{lbl_lower}"] = float(p[idx])
        metrics[f"recall_{lbl_lower}"] = float(r[idx])

    return metrics


def evaluate_transformer(
    model_name: str, test_df: pd.DataFrame, output_prefix: str, batch_size: int = 32
) -> dict[str, Any]:
    """Ejecuta inferencia por batches en CPU para evitar cuelgues."""
    model_dir = PROJ_ROOT / "models" / "encoders" / model_name
    if not model_dir.exists():
        print(f"Skipping {model_name}: Directory {model_dir} not found.")
        return {}

    print(f"\n[Evaluando Transformer: {model_name}]")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to("cpu")
    model.eval()

    input_type = "command-context" if "context" in model_name else "command"
    text_inputs: list[str] = [
        build_input(row, input_type) for row in test_df.to_dict(orient="records")
    ]

    all_preds = []
    total_samples = len(text_inputs)

    # Inferencia por lotes para no saturar memoria
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_texts = text_inputs[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cpu")

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

            print(
                f" - Procesados {min(i + batch_size, total_samples)}/{total_samples} ejemplos...",
                end="\r",
            )

    print("\n - Inferencia completada. Mapeando etiquetas...")

    # Mapeo de ID a Label desde config del modelo
    id2label = getattr(model.config, "id2label", {0: "ALLOW", 1: "DENY", 2: "ASK"})
    predicted_labels = [id2label.get(int(p), str(p)) for p in all_preds]

    res_df = (
        test_df[["id", "label"]].copy() if "label" in test_df.columns else test_df[["id"]].copy()
    )
    res_df.rename(columns={"label": "true_label"}, inplace=True)
    res_df["predicted_label"] = predicted_labels

    metrics = compute_metrics(res_df)

    preds_path = model_dir / f"{output_prefix}_predictions.jsonl"
    metrics_path = model_dir / f"{output_prefix}_metrics.json"

    res_df.to_json(preds_path, orient="records", lines=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f" [{model_name}] Guardado en: {metrics_path}")
    return metrics


def evaluate_baseline(test_df: pd.DataFrame, output_prefix: str) -> dict[str, Any]:
    """Ejecuta inferencia con baseline Scikit-Learn."""
    model_dir = PROJ_ROOT / "models" / "tfidf_char_linear_svm"
    pipeline_path = model_dir / "pipeline.joblib"
    if not pipeline_path.exists():
        print(f"\nSkipping Baseline: {pipeline_path} not found.")
        return {}

    print("\n[Evaluando Baseline: tfidf_char_linear_svm]")
    pipeline = joblib.load(pipeline_path)

    text_inputs: list[str] = [
        build_input(row, "command") for row in test_df.to_dict(orient="records")
    ]
    preds = pipeline.predict(text_inputs)

    res_df = (
        test_df[["id", "label"]].copy() if "label" in test_df.columns else test_df[["id"]].copy()
    )
    res_df.rename(columns={"label": "true_label"}, inplace=True)
    res_df["predicted_label"] = preds

    metrics = compute_metrics(res_df)

    preds_path = model_dir / f"{output_prefix}_predictions.jsonl"
    metrics_path = model_dir / f"{output_prefix}_metrics.json"

    res_df.to_json(preds_path, orient="records", lines=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f" [Baseline] Guardado en: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de modelos congelados")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/shell_safety/test.jsonl",
        help="Ruta al archivo JSONL de evaluación",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="test",
        help="Prefijo para archivos exportados (ej. 'test' o 'fixture')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de lote para la inferencia en CPU",
    )
    args = parser.parse_args()

    data_file = Path(args.data_path)
    if not data_file.is_absolute():
        data_file = PROJ_ROOT / data_file

    print(f"Cargando dataset desde: {data_file}")
    test_df = pd.read_json(data_file, lines=True)

    for model_name in MODEL_NAMES:
        evaluate_transformer(model_name, test_df, args.output_prefix, args.batch_size)

    evaluate_baseline(test_df, args.output_prefix)


if __name__ == "__main__":
    main()
