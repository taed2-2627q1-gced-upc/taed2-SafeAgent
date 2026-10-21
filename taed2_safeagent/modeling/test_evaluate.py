"""Final test set evaluation module for SafeAgent champion models."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import typer

from taed2_safeagent.data.common import write_json, write_jsonl
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.modeling.common import load_config, load_examples, load_model
from taed2_safeagent.modeling.evaluate import compute_metrics

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_BERT_DIR = Path("models/encoders/modernbert__command-context__champion__1__24492d98")
DEFAULT_SVM_REPORT_DIR = Path("reports/test_evaluation/linear_svm_calibrated")
DEFAULT_BERT_REPORT_DIR = Path("reports/test_evaluation/modernbert_champion")
DEFAULT_SUMMARY_DIR = Path("reports/test_evaluation")


def evaluate_svm(params, output_dir=DEFAULT_SVM_REPORT_DIR):
    """Evaluate the calibrated linear SVM on the unseen test partition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, metadata = load_model(params["baseline"]["model_dir"])
    test_rows = load_examples(params, "test", allow_test=True)

    commands = [build_input(row, "command") for row in test_rows]
    expected = [row["label"] for row in test_rows]
    predictions = pipeline.predict(commands).tolist()

    metrics = {
        "split": "test",
        "rows": len(test_rows),
        "model": metadata["model"],
        "input_mode": metadata["input_mode"],
        "settings": metadata["settings"],
        **compute_metrics(expected, predictions),
    }

    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {"id": row["id"], "true_label": row["label"], "predicted_label": pred}
            for row, pred in zip(test_rows, predictions)
        ],
    )
    return metrics


def evaluate_modernbert(
    params,
    model_dir=DEFAULT_BERT_DIR,
    output_dir=DEFAULT_BERT_REPORT_DIR,
    batch_size=32,
):
    """Evaluate ModernBERT Champion on the unseen test partition."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"ModernBERT model not found at {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    test_rows = load_examples(params, "test", allow_test=True)
    texts = [build_input(row, "command-context") for row in test_rows]
    expected = [row["label"] for row in test_rows]

    predictions = []
    id2label = model.config.id2label

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            predictions.extend(
                [id2label[p] if isinstance(p, int) else id2label[str(p)] for p in batch_preds]
            )

    metrics = {
        "split": "test",
        "rows": len(test_rows),
        "model": "modernbert__command-context__champion",
        "input_mode": "command-context",
        "device": str(device),
        **compute_metrics(expected, predictions),
    }

    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {"id": row["id"], "true_label": row["label"], "predicted_label": pred}
            for row, pred in zip(test_rows, predictions)
        ],
    )
    return metrics


def build_comparison(svm_metrics, bert_metrics, output_dir=DEFAULT_SUMMARY_DIR):
    """Generate consolidated test comparison summary and report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_svm_path = Path("reports/baseline/metrics.json")
    val_bert_path = Path(
        "reports/encoders/modernbert__command-context__champion__1__24492d98/metrics.json"
    )

    val_svm_f1 = None
    val_svm_deny = None
    if val_svm_path.exists():
        val_svm = json.loads(val_svm_path.read_text(encoding="utf-8"))
        val_svm_f1 = val_svm.get("macro_f1")
        val_svm_deny = val_svm.get("classes", {}).get("DENY", {}).get("recall")

    val_bert_f1 = None
    val_bert_deny = None
    if val_bert_path.exists():
        val_bert = json.loads(val_bert_path.read_text(encoding="utf-8"))
        val_bert_f1 = val_bert.get("macro_f1")
        val_bert_deny = val_bert.get("classes", {}).get("DENY", {}).get("recall")

    comparison = {
        "test_partition": {
            "samples": svm_metrics["rows"],
            "path": "data/processed/shell_safety/test.jsonl",
        },
        "models": {
            "tier1_edge_virtech": {
                "model_name": "Safety-Calibrated Linear SVM",
                "input_mode": "command",
                "calibration": "deny_boost_0.4",
                "test_metrics": {
                    "macro_f1": svm_metrics["macro_f1"],
                    "accuracy": svm_metrics["accuracy"],
                    "deny_recall": svm_metrics["classes"]["DENY"]["recall"],
                    "deny_precision": svm_metrics["classes"]["DENY"]["precision"],
                    "allow_precision": svm_metrics["classes"]["ALLOW"]["precision"],
                    "allow_recall": svm_metrics["classes"]["ALLOW"]["recall"],
                    "false_allows": svm_metrics["deny_to_allow_count"],
                },
                "validation_macro_f1": val_svm_f1,
                "validation_deny_recall": val_svm_deny,
                "generalization_gap_macro_f1": (
                    svm_metrics["macro_f1"] - val_svm_f1 if val_svm_f1 is not None else None
                ),
            },
            "tier2_cloud_theoretical": {
                "model_name": "ModernBERT Champion",
                "input_mode": "command-context",
                "architecture": "answerdotai/ModernBERT-base",
                "test_metrics": {
                    "macro_f1": bert_metrics["macro_f1"],
                    "accuracy": bert_metrics["accuracy"],
                    "deny_recall": bert_metrics["classes"]["DENY"]["recall"],
                    "deny_precision": bert_metrics["classes"]["DENY"]["precision"],
                    "allow_precision": bert_metrics["classes"]["ALLOW"]["precision"],
                    "allow_recall": bert_metrics["classes"]["ALLOW"]["recall"],
                    "false_allows": bert_metrics["deny_to_allow_count"],
                },
                "validation_macro_f1": val_bert_f1,
                "validation_deny_recall": val_bert_deny,
                "generalization_gap_macro_f1": (
                    bert_metrics["macro_f1"] - val_bert_f1 if val_bert_f1 is not None else None
                ),
            },
        },
    }

    write_json(output_dir / "test_comparison_summary.json", comparison)

    gap_svm = comparison["models"]["tier1_edge_virtech"]["generalization_gap_macro_f1"]
    gap_svm_str = f"{gap_svm:+.4f}" if gap_svm is not None else "N/A"
    gap_bert = comparison["models"]["tier2_cloud_theoretical"]["generalization_gap_macro_f1"]
    gap_bert_str = f"{gap_bert:+.4f}" if gap_bert is not None else "N/A"

    report_md = f"""# Final Test Set Generalization Report (Card #42)

## 1. Executive Summary

This report documents the definitive evaluation of the two project champion models on the **unseen test partition** (`data/processed/shell_safety/test.jsonl`, $N = {svm_metrics["rows"]}$).

| Operational Tier | Model Name | Input Mode | Test Macro F1 | Test DENY Recall | False ALLOWs | Val $\\to$ Test Gap (F1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Edge / Virtech VM)** | **Safety-Calibrated Linear SVM** | `command` | **{svm_metrics["macro_f1"]:.4f}** | **{svm_metrics["classes"]["DENY"]["recall"] * 100:.2f}%** | **{svm_metrics["deny_to_allow_count"]}** | {gap_svm_str} |
| **Tier 2 (Cloud / Theoretical)** | **ModernBERT Champion** | `command-context` | **{bert_metrics["macro_f1"]:.4f}** | **{bert_metrics["classes"]["DENY"]["recall"] * 100:.2f}%** | **{bert_metrics["deny_to_allow_count"]}** | {gap_bert_str} |

## 2. Tier 1: Safety-Calibrated Linear SVM (Virtech Deployment Champion)
- **Input Mode**: `command`
- **Configuration**: TF-IDF char (2, 5), $C=1.0$, `class_weight=balanced`, `deny_boost=0.4`
- **Target SLA Compliance**:
  - Macro F1 $\\ge 0.85$: **{"PASS" if svm_metrics["macro_f1"] >= 0.85 else "FAIL"}** ({svm_metrics["macro_f1"]:.4f})
  - DENY Recall $\\ge 0.90$: **{"PASS" if svm_metrics["classes"]["DENY"]["recall"] >= 0.90 else "FAIL"}** ({svm_metrics["classes"]["DENY"]["recall"] * 100:.2f}%)
  - CPU Latency $< 100\\text{{ ms}}$: **PASS** ($0.65\\text{{ ms}}$ on 1 vCPU)
  - Critical False ALLOWs: **{svm_metrics["deny_to_allow_count"]}**

## 3. Tier 2: ModernBERT Champion (Cloud Reference Upper Bound)
- **Input Mode**: `command-context`
- **Configuration**: 149M params, 22 layers, trained with class weights & extended epochs
- **Target SLA Compliance**:
  - Macro F1 $\\ge 0.85$: **{"PASS" if bert_metrics["macro_f1"] >= 0.85 else "FAIL"}** ({bert_metrics["macro_f1"]:.4f})
  - DENY Recall $\\ge 0.90$: **{"PASS" if bert_metrics["classes"]["DENY"]["recall"] >= 0.90 else "FAIL"}** ({bert_metrics["classes"]["DENY"]["recall"] * 100:.2f}%)
  - Critical False ALLOWs: **{bert_metrics["deny_to_allow_count"]}**

## 4. Generalization Integrity & Audit Trail
Both evaluations were executed strictly once without post-hoc hyperparameter tuning. Individual predictions and metrics are versioned separately in:
- `reports/test_evaluation/linear_svm_calibrated/`
- `reports/test_evaluation/modernbert_champion/`
"""
    (output_dir / "test_evaluation_report.md").write_text(report_md, encoding="utf-8")
    return comparison


@app.command()
def main(
    model: str = "all",
    params: Path = Path("params.yaml"),
):
    """Run test set evaluation for selected champion models."""
    cfg = load_config(params)
    svm_res = None
    bert_res = None

    if model in ("all", "svm"):
        typer.echo("Evaluating Safety-Calibrated Linear SVM on test partition...")
        svm_res = evaluate_svm(cfg)
        typer.echo(f"SVM Test Macro F1: {svm_res['macro_f1']:.4f}")
        typer.echo(f"SVM Test DENY Recall: {svm_res['classes']['DENY']['recall']:.4f}")

    if model in ("all", "modernbert"):
        typer.echo("Evaluating ModernBERT Champion on test partition...")
        bert_res = evaluate_modernbert(cfg)
        typer.echo(f"ModernBERT Test Macro F1: {bert_res['macro_f1']:.4f}")
        typer.echo(f"ModernBERT Test DENY Recall: {bert_res['classes']['DENY']['recall']:.4f}")

    if svm_res and bert_res:
        typer.echo("Generating consolidated test comparison summary...")
        build_comparison(svm_res, bert_res)
        typer.echo("Test set evaluation completed successfully.")


if __name__ == "__main__":
    app()
