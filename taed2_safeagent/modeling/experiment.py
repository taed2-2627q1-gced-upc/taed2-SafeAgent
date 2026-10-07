from pathlib import Path

from mlflow.exceptions import MlflowException
import typer

from taed2_safeagent.data.common import write_json
from taed2_safeagent.modeling.common import load_config
from taed2_safeagent.modeling.evaluate import evaluate
from taed2_safeagent.modeling.tracking import (
    check_inputs,
    get_experiment_id,
    log_input_lock,
    log_results,
    mark_failed,
    save_model_version,
    tracking_client,
)
from taed2_safeagent.modeling.train import train

app = typer.Typer(pretty_exceptions_show_locals=False)


def run_experiment(params, root):
    root = Path(root).resolve()
    client = tracking_client(params["tracking"])
    source, lock_version = check_inputs(root)
    input_lock = (root / "dvc.lock").read_bytes()
    seed = params["baseline"]["svm"]["random_state"]
    name = f"tfidf_char_linear_svm__command__seed{seed}"
    tags = {
        "git_commit_sha": source["commit"],
        "git_is_dirty": "false",
        "mlflow.source.git.commit": source["commit"],
        "mlflow.source.git.repoURL": "https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent",
        "input_dvc_lock_hash": lock_version,
        "model_type": "tfidf_char_linear_svm",
        "input_mode": "command",
        "seed": str(seed),
        "evaluation_split": "validation",
    }
    try:
        experiment_id = get_experiment_id(client, params["tracking"]["experiment"])
        run = client.create_run(experiment_id, run_name=name, tags=tags)
    except MlflowException:
        raise RuntimeError("Could not start the shared experiment") from None
    receipt = {
        "run_id": run.info.run_id,
        "experiment_id": experiment_id,
        "tracking_uri": params["tracking"]["uri"].rstrip("/"),
        "source_commit": source["commit"],
        "input_dvc_lock_hash": lock_version,
    }
    receipt["run_url"] = (
        f"{receipt['tracking_uri']}/#/experiments/{experiment_id}/runs/{receipt['run_id']}"
    )
    report = params["baseline"]["report_dir"] / "mlflow_run.json"
    phase = "training"
    try:
        typer.echo("Training the baseline")
        metadata = train(params)
        report.unlink(missing_ok=True)
        if metadata["code"] != source:
            raise ValueError("Source changed during training")
        phase = "validation"
        metadata["tracking"] = {
            key: receipt[key] for key in ("run_id", "experiment_id", "tracking_uri", "run_url")
        }
        write_json(params["baseline"]["model_dir"] / "metadata.json", metadata)
        metrics = evaluate(params)
        phase = "model upload"
        typer.echo("Saving the model with DVC")
        lock, model_version = save_model_version(root, params["baseline"]["model_dir"])
        phase = "result logging"
        typer.echo("Recording the validation results")
        client.set_tag(receipt["run_id"], "model_dvc_version", model_version, synchronous=True)
        client.set_tag(
            receipt["run_id"],
            "model_dvc_path",
            lock.relative_to(root).as_posix(),
            synchronous=True,
        )
        client.set_tag(receipt["run_id"], "model_dvc_stage", "train", synchronous=True)
        log_input_lock(client, receipt["run_id"], input_lock)
        files = (
            params["baseline"]["model_dir"] / "metadata.json",
            lock,
            params["baseline"]["report_dir"] / "metrics.json",
            params["baseline"]["report_dir"] / "predictions.jsonl",
        )
        log_results(client, receipt["run_id"], files, metadata, metrics)
        phase = "run completion"
        client.set_terminated(receipt["run_id"], status="FINISHED")
        if client.get_run(receipt["run_id"]).info.status != "FINISHED":
            raise ValueError("Shared run is not finished")
        receipt.update(
            status="FINISHED",
            model_dvc_version=model_version,
            model_dvc_path=lock.relative_to(root).as_posix(),
            model_dvc_stage="train",
        )
        write_json(report, receipt)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, MlflowException):
        failed = mark_failed(client, receipt["run_id"])
        state = "marked failed" if failed else "not confirmed, check DagsHub"
        raise RuntimeError(
            f"Experiment failed during {phase}. Run {receipt['run_id']}: {state}"
        ) from None
    return receipt


@app.command()
def main(params: Path = Path("params.yaml")):
    try:
        receipt = run_experiment(load_config(params), params.resolve().parent)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, MlflowException) as error:
        message = "Tracking request failed" if isinstance(error, MlflowException) else str(error)
        typer.echo(f"Experiment failed: {message}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Run saved: {receipt['run_url']}")


if __name__ == "__main__":
    app()
