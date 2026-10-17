import json
from pathlib import Path
import random
from typing import Any

from mlflow.exceptions import MlflowException
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import typer

from taed2_safeagent.data.common import (
    LABELS,
    load_params,
    read_jsonl,
    write_json,
    write_jsonl,
)
from taed2_safeagent.data.inputs import build_input, safe_context
from taed2_safeagent.modeling.common import load_examples
from taed2_safeagent.modeling.evaluate import compute_metrics
from taed2_safeagent.modeling.tracking import (
    evaluation_metric_values,
    get_experiment_id,
    tracking_client,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

EMPTY_CONTEXT: dict[str, Any] = {
    "gitRemote": None,
    "gitStatus": {"untracked": [], "modified": [], "staged": []},
    "agentTouchedFiles": [],
    "lastUserPrompt": None,
    "cwd": None,
}


def make_shuffled_examples(examples: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Permute execution contexts across rows while keeping commands and labels unchanged."""
    if not examples:
        raise ValueError("Cannot shuffle empty examples")
    contexts = [ex["context"] for ex in examples]
    rng = random.Random(seed)
    indices = list(range(len(contexts)))
    rng.shuffle(indices)

    shuffled = []
    for original, permuted_idx in zip(examples, indices):
        shuffled.append(
            {
                "id": original["id"],
                "command": original["command"],
                "context": safe_context(contexts[permuted_idx]),
                "label": original["label"],
            }
        )
    return shuffled


def make_empty_context_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace each row's context with the minimal valid empty context."""
    if not examples:
        raise ValueError("Cannot create empty context for empty examples")
    empty_ctx = safe_context(EMPTY_CONTEXT)
    return [
        {
            "id": ex["id"],
            "command": ex["command"],
            "context": empty_ctx,
            "label": ex["label"],
        }
        for ex in examples
    ]


def predict_encoder_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> list[str]:
    """Run batched inference on rows using the command-context input format."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels: list[str] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        texts = [build_input(row, "command-context") for row in chunk]
        batch = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            logits = model(**batch).logits
            if not torch.isfinite(logits).all():
                raise ValueError("Encountered non-finite logits during inference")
            pred_indices = logits.argmax(dim=-1).tolist()
            all_labels.extend(LABELS[idx] for idx in pred_indices)

    return all_labels


def evaluate_ablation(
    model_dir: Path,
    mode: str,
    examples: list[dict[str, Any]],
    seed: int = 42,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run context ablation evaluation (shuffled or empty) for a context-trained model."""
    model_dir = Path(model_dir).resolve()
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {model_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("input_mode") != "command-context":
        raise ValueError(
            f"Context ablation is only applicable to 'command-context' models. "
            f"Found input_mode: '{metadata.get('input_mode')}'"
        )

    # Prepare inputs according to ablation mode
    if mode == "shuffled":
        ablated_rows = make_shuffled_examples(examples, seed=seed)
    elif mode == "empty":
        ablated_rows = make_empty_context_examples(examples)
    else:
        raise ValueError(f"Unknown ablation mode '{mode}'. Choose 'shuffled' or 'empty'.")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )

    max_length = metadata["settings"].get("max_length", 512)
    predictions = predict_encoder_batch(
        model, tokenizer, ablated_rows, max_length=max_length, batch_size=batch_size, device=device
    )

    expected = [row["label"] for row in examples]
    metrics = compute_metrics(expected, predictions)

    # Baseline comparison metrics
    baseline_f1 = metadata.get("best_validation_macro_f1")
    delta_f1 = (metrics["macro_f1"] - baseline_f1) if baseline_f1 is not None else None

    # Load original baseline validation predictions if available for delta DENY comparison
    val_pred_file = model_dir / "validation_predictions.jsonl"
    baseline_deny_recall = None
    baseline_false_allow = None
    if val_pred_file.exists():
        original_preds = [row["predicted_label"] for _, row in read_jsonl(val_pred_file)]
        orig_metrics = compute_metrics(expected, original_preds)
        baseline_deny_recall = orig_metrics["classes"]["DENY"]["recall"]
        baseline_false_allow = orig_metrics["deny_to_allow_count"]

    delta_deny_recall = (
        metrics["classes"]["DENY"]["recall"] - baseline_deny_recall
        if baseline_deny_recall is not None
        else None
    )
    delta_false_allow = (
        metrics["deny_to_allow_count"] - baseline_false_allow
        if baseline_false_allow is not None
        else None
    )

    result = {
        "model": metadata.get("model"),
        "model_dir": str(model_dir),
        "source_checkpoint": metadata.get("code", {}).get("commit"),
        "ablation_mode": mode,
        "seed": seed if mode == "shuffled" else None,
        "rows": len(examples),
        "metrics": metrics,
        "baseline_comparison": {
            "baseline_macro_f1": baseline_f1,
            "delta_macro_f1": delta_f1,
            "baseline_deny_recall": baseline_deny_recall,
            "delta_deny_recall": delta_deny_recall,
            "baseline_deny_to_allow_count": baseline_false_allow,
            "delta_deny_to_allow_count": delta_false_allow,
        },
        "predictions": [
            {
                "id": row["id"],
                "command": row["command"],
                "true_label": row["label"],
                "predicted_label": pred,
            }
            for row, pred in zip(examples, predictions)
        ],
    }
    return result


def log_to_mlflow_if_available(result: dict[str, Any], params: dict[str, Any]) -> str | None:
    """Log ablation experiment to shared DagsHub MLflow if credentials are set."""
    try:
        tracking_settings = params.get("tracking", {})
        client = tracking_client(tracking_settings)
        experiment_id = get_experiment_id(client, tracking_settings["experiment"])

        model_name = result["model"]
        mode = result["ablation_mode"]
        run_name = f"{model_name}__ablation-{mode}__seed{result['seed'] or 42}"

        from mlflow.entities import Param

        run = client.create_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={
                "experiment_type": "context_ablation",
                "ablation_mode": mode,
                "model_type": model_name,
                "base_model": Path(result["model_dir"]).name,
            },
        )

        # Log parameters
        params_to_log = [
            Param("ablation_mode", mode),
            Param("model", model_name),
            Param("seed", str(result["seed"] or 42)),
            Param("rows", str(result["rows"])),
        ]
        for p in params_to_log:
            client.log_param(run.info.run_id, p.key, p.value)

        # Log metrics
        metric_dict = evaluation_metric_values(result["metrics"])
        comp = result["baseline_comparison"]
        if comp["delta_macro_f1"] is not None:
            metric_dict["delta_macro_f1"] = comp["delta_macro_f1"]
        if comp["delta_deny_recall"] is not None:
            metric_dict["delta_deny_recall"] = comp["delta_deny_recall"]

        for key, val in metric_dict.items():
            client.log_metric(run.info.run_id, key, val)

        client.set_terminated(run.info.run_id, status="FINISHED")
        return run.info.run_id
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, MlflowException) as error:
        typer.echo(f"Notice: MLflow logging skipped ({error})", err=True)
        return None


@app.command()
def main(
    model_dir: Path,
    mode: str = "both",
    seed: int = 42,
    batch_size: int = 32,
    output_dir: Path = Path("reports/context_experiments"),
    params_file: Path = Path("params.yaml"),
    mlflow_log: bool = True,
):
    """Run context ablation experiments (shuffled context and/or empty context) on a context model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_params(params_file)
    examples = load_examples(config, "validation")

    modes = ["shuffled", "empty"] if mode == "both" else [mode]
    model_name = Path(model_dir).name.split("__")[0]

    for m in modes:
        typer.echo("\n=======================================================")
        typer.echo(f"Running {m.upper()} context ablation for {model_dir.name}...")
        typer.echo("=======================================================")
        res = evaluate_ablation(
            model_dir=model_dir,
            mode=m,
            examples=examples,
            seed=seed,
            batch_size=batch_size,
        )

        # Save metrics JSON (without the full predictions array to keep file lightweight)
        report_data = {key: val for key, val in res.items() if key != "predictions"}
        out_file = output_dir / f"{model_name}__{m}_context.json"
        write_json(out_file, report_data)
        typer.echo(f"Saved metric report to {out_file}")

        # Save predictions
        pred_file = output_dir / f"{model_name}__{m}_predictions.jsonl"
        write_jsonl(pred_file, res["predictions"])
        typer.echo(f"Saved prediction records to {pred_file}")

        # Console Summary
        typer.echo(f"--- Results Summary for {model_name} ({m}) ---")
        delta_f1 = res["baseline_comparison"]["delta_macro_f1"]
        delta_dr = res["baseline_comparison"]["delta_deny_recall"]
        delta_d2a = res["baseline_comparison"]["delta_deny_to_allow_count"]
        typer.echo(
            f"Macro F1:            {res['metrics']['macro_f1']:.4f} "
            f"({('Δ ' + f'{delta_f1:+.4f}') if delta_f1 is not None else 'N/A'})"
        )
        typer.echo(
            f"DENY Recall:         {res['metrics']['classes']['DENY']['recall']:.4f} "
            f"({('Δ ' + f'{delta_dr:+.4f}') if delta_dr is not None else 'N/A'})"
        )
        typer.echo(
            f"DENY -> ALLOW Count: {res['metrics']['deny_to_allow_count']} "
            f"({('Δ ' + f'{delta_d2a:+d}') if delta_d2a is not None else 'N/A'})"
        )

        if mlflow_log:
            run_id = log_to_mlflow_if_available(res, config)
            if run_id:
                typer.echo(f"Logged to MLflow run ID: {run_id}")


if __name__ == "__main__":
    app()
