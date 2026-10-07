import hashlib
import json
from pathlib import Path
import re
import shutil

from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
import typer
import yaml

from taed2_safeagent.data.common import LABELS, load_params, read_jsonl, write_json
from taed2_safeagent.modeling.common import code_identity, dataset_identity, load_examples
from taed2_safeagent.modeling.evaluate import compute_metrics
from taed2_safeagent.modeling.tracking import (
    dvc_command,
    evaluation_metric_values,
    get_experiment_id,
    mark_failed,
    tracking_client,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


def checked_model(result, metadata):
    config = json.loads((result / "model/config.json").read_text(encoding="utf-8"))
    if config["id2label"] != {str(index): label for index, label in enumerate(LABELS)}:
        raise ValueError("Saved model label mapping differs")
    expected = {"codebert": "roberta", "modernbert": "modernbert"}
    if config["model_type"] != expected[metadata["model"]]:
        raise ValueError("Saved model architecture differs")
    files = list((result / "model").rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("Model files must be regular files")
    size = sum(path.stat().st_size for path in files if path.is_file())
    if size != metadata["model_bytes"] or not list((result / "model").glob("*.safetensors")):
        raise ValueError("Saved model files differ from the recorded artifact")


def checked_result(result, params):
    result = Path(result).resolve()
    metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    source = metadata["code"]
    if metadata["status"] != "FINISHED" or source["dirty"] is not False:
        raise ValueError("Only completed runs from committed source can be registered")
    if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
        raise ValueError("Missing training source version")
    if metadata["labels"] != list(LABELS) or metadata["input_mode"] not in {
        "command",
        "command-context",
    }:
        raise ValueError("Unexpected model labels or input mode")
    if metadata["backbone"] != params["encoders"]["backbones"][metadata["model"]]:
        raise ValueError("Backbone differs from the approved configuration")
    if metadata["settings"] != params["encoders"]["training"]:
        raise ValueError("Training settings differ from the approved configuration")
    if metadata["dataset"] != dataset_identity(params):
        raise ValueError("Training data differs from the prepared dataset")
    lock = (result / "input_dvc.lock").read_bytes()
    if hashlib.sha256(lock).hexdigest() != metadata["input_dvc_lock_hash"]:
        raise ValueError("Input data version file differs")
    if metadata["end_time_ms"] <= metadata["start_time_ms"]:
        raise ValueError("Invalid training timestamps")
    rows = load_examples(params, "validation")
    predictions = [row for _, row in read_jsonl(result / "predictions.jsonl")]
    if [row["id"] for row in predictions] != [row["id"] for row in rows]:
        raise ValueError("Validation predictions are not aligned")
    expected = [row["label"] for row in rows]
    if [row["true_label"] for row in predictions] != expected:
        raise ValueError("Validation targets differ from the dataset")
    actual = {
        "split": "validation",
        "rows": len(rows),
        **compute_metrics(expected, [row["predicted_label"] for row in predictions]),
    }
    if metrics != actual:
        raise ValueError("Saved metrics differ from the predictions")
    checked_model(result, metadata)
    return metadata, metrics


def shared_metrics(metadata, metrics, energy):
    values = evaluation_metric_values(metrics)
    for key in ("training_seconds", "parameters", "model_bytes"):
        values[key] = metadata[key]
    if energy["status"] == "recorded":
        measurement = energy["measurement"]
        values.update(
            energy_kwh=measurement["energy_consumed"],
            emissions_kg=measurement["emissions"],
            measurement_seconds=measurement["duration"],
        )
    return values


def log_registered_result(client, run_id, metadata, metrics, files):
    energy = json.loads(
        next(file for file in files if file.name == "energy.json").read_text(encoding="utf-8")
    )
    settings = {
        **metadata["settings"],
        "backbone": metadata["backbone"]["name"],
        "backbone_revision": metadata["backbone"]["revision"],
        "source_revision": metadata["dataset"]["source"]["revision"],
        "energy_country_assumed": energy["country_assumed"],
    }
    client.log_batch(
        run_id,
        params=[Param(key, str(value)) for key, value in settings.items()],
        metrics=[
            Metric(key, value, metadata["end_time_ms"], 0)
            for key, value in shared_metrics(metadata, metrics, energy).items()
        ],
        synchronous=True,
    )
    for file in files:
        client.log_artifact(run_id, str(file))
    client.log_dict(run_id, metrics["classes"], "classification_report.json")
    matrix = metrics["confusion_matrix"]
    client.log_dict(run_id, matrix, "confusion_matrix.json")
    normalized = {
        **matrix,
        "counts": [
            [value / sum(row) if sum(row) else 0 for value in row] for row in matrix["counts"]
        ],
    }
    client.log_dict(run_id, normalized, "confusion_matrix_normalized.json")


def checked_session(session_path, metadata):
    session = json.loads(Path(session_path).read_text(encoding="utf-8"))
    if session["git_commit"] != metadata["code"]["commit"]:
        raise ValueError("Cloud session and training source differ")
    if session["saved_version_status"] != "confirmed":
        raise ValueError("Confirm the saved Kaggle version before registration")
    if not re.fullmatch(r"[a-z0-9_]+/[a-z0-9_-]+", session["kaggle_ref"]):
        raise ValueError("Invalid Kaggle session reference")
    if (
        not isinstance(session["kaggle_version"], int)
        or isinstance(session["kaggle_version"], bool)
        or session["kaggle_version"] < 1
    ):
        raise ValueError("Invalid Kaggle source version")
    return session


def register_result(params_path, result, session_path):
    params_path = Path(params_path).resolve()
    result = Path(result).resolve()
    root = params_path.parent
    params = load_params(params_path)
    metadata, metrics = checked_result(result, params)
    session = checked_session(session_path, metadata)
    registration = code_identity()
    if not registration["commit"] or registration["dirty"] is not False:
        raise ValueError("Commit registration code before uploading results")
    name = f"{metadata['model']}__{metadata['input_mode']}__{session['kaggle_version']}"
    name += f"__{metadata['code']['commit'][:8]}"
    target = root / "models/encoders" / name
    report = root / "reports/encoders" / name
    if target.exists() or report.exists():
        raise ValueError("This result already exists, inspect it before another registration")
    client = tracking_client(params["tracking"])
    experiment_id = get_experiment_id(client, params["tracking"]["experiment"])
    run = client.create_run(
        experiment_id,
        start_time=metadata["start_time_ms"],
        run_name=(
            f"{metadata['model']}__{metadata['input_mode']}__seed{metadata['settings']['seed']}"
        ),
        tags={
            "git_commit_sha": metadata["code"]["commit"],
            "git_is_dirty": "false",
            "mlflow.source.git.commit": metadata["code"]["commit"],
            "input_dvc_lock_hash": metadata["input_dvc_lock_hash"],
            "model_type": metadata["model"],
            "input_mode": metadata["input_mode"],
            "evaluation_split": "validation",
            "kaggle_ref": session["kaggle_ref"],
            "kaggle_version": str(session["kaggle_version"]),
            "registration_commit": registration["commit"],
        },
    )
    run_id = run.info.run_id
    try:
        metadata["tracking"] = {"run_id": run_id, "experiment_id": experiment_id}
        metadata["cloud"] = {
            key: session[key] for key in ("kaggle_ref", "kaggle_version", "git_commit")
        }
        shutil.copytree(result / "model", target)
        write_json(target / "metadata.json", metadata)
        shutil.copyfile(result / "predictions.jsonl", target / "validation_predictions.jsonl")
        report.mkdir(parents=True)
        for filename in ("metrics.json", "energy.json", "input_dvc.lock"):
            shutil.copyfile(result / filename, report / filename)
        relative = target.relative_to(root).as_posix()
        dvc_command(root, "add", relative)
        pointer = Path(f"{target}.dvc")
        model_version = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]
        dvc_command(root, "push", pointer.relative_to(root).as_posix())
        client.set_tag(run_id, "model_dvc_version", model_version, synchronous=True)
        client.set_tag(
            run_id, "model_dvc_path", pointer.relative_to(root).as_posix(), synchronous=True
        )
        log_registered_result(
            client,
            run_id,
            metadata,
            metrics,
            (
                target / "metadata.json",
                pointer,
                target / "validation_predictions.jsonl",
                *report.iterdir(),
            ),
        )
        client.set_terminated(run_id, end_time=metadata["end_time_ms"], status="FINISHED")
        if client.get_run(run_id).info.status != "FINISHED":
            raise ValueError("Shared run completion is not confirmed")
        receipt = {
            "run_id": run_id,
            "status": "FINISHED",
            "source_commit": metadata["code"]["commit"],
            "registration_commit": registration["commit"],
            "model_dvc_version": model_version,
            "model_dvc_path": pointer.relative_to(root).as_posix(),
            "kaggle_ref": session["kaggle_ref"],
            "kaggle_version": session["kaggle_version"],
            "run_url": f"{params['tracking']['uri']}/#/experiments/{experiment_id}/runs/{run_id}",
        }
        write_json(report / "mlflow_run.json", receipt)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, MlflowException):
        mark_failed(client, run_id)
        raise RuntimeError(
            f"Could not register run {run_id}, inspect the retained files"
        ) from None
    return receipt


@app.command()
def main(result: Path, session: Path, params: Path = Path("params.yaml")):
    try:
        receipt = register_result(params, result, session)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, MlflowException) as error:
        message = type(error).__name__ if isinstance(error, MlflowException) else str(error)
        typer.echo(f"Result registration failed: {message}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Run saved: {receipt['run_url']}")


if __name__ == "__main__":
    app()
