import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import time
from urllib.parse import urlsplit

from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
import yaml

from taed2_safeagent.modeling.common import code_identity


def tracking_client(settings):
    uri = settings["uri"].rstrip("/")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "dagshub.com"
        or not parsed.path.endswith(".mlflow")
    ):
        raise ValueError("Use a DagsHub tracking URI")
    if parsed.username is not None or parsed.query or parsed.fragment:
        raise ValueError("Use a DagsHub tracking URI without credentials")
    for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        if not os.environ.get(name):
            raise ValueError(f"Missing environment variable: {name}")
    configured = os.environ.get("MLFLOW_TRACKING_URI", uri).rstrip("/")
    if configured != uri:
        raise ValueError("Tracking URI differs from params.yaml")
    if not settings["experiment"].strip():
        raise ValueError("Missing experiment name")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
    return MlflowClient(tracking_uri=uri)


def dvc_command(root, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "dvc", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"DVC {arguments[0]} failed")
    return result.stdout


def check_inputs(root):
    source = code_identity()
    if not source["commit"] or source["dirty"] is not False:
        raise ValueError("Commit the source and settings before a tracked run")
    if json.loads(dvc_command(root, "status", "--json", "audit", "prepare", "validate")):
        raise ValueError("DVC inputs have changed, prepare them before training")
    lock = (root / "dvc.lock").read_bytes()
    return source, hashlib.sha256(lock).hexdigest()


def save_model_version(root, directory):
    relative = directory.relative_to(root).as_posix()
    dvc_command(root, "commit", "--force", "train", "evaluate")
    lock = root / "dvc.lock"
    outputs = yaml.safe_load(lock.read_text(encoding="utf-8"))["stages"]["train"]["outs"]
    versions = [output["md5"] for output in outputs if output["path"] == relative]
    if len(versions) != 1 or not versions[0].endswith(".dir"):
        raise ValueError("Missing trained model version in dvc.lock")
    dvc_command(root, "push", "train")
    return lock, versions[0]


def log_input_lock(client, run_id, content):
    with TemporaryDirectory(prefix="safeagent-input-") as directory:
        lock = Path(directory) / "input_dvc.lock"
        lock.write_bytes(content)
        client.log_artifact(run_id, str(lock))


def get_experiment_id(client, name):
    experiment = client.get_experiment_by_name(name)
    return experiment.experiment_id if experiment else client.create_experiment(name)


def evaluation_metric_values(metrics):
    values = {
        key: metrics[key]
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "deny_to_allow_count",
            "deny_to_allow_rate",
        )
    }
    for label, scores in metrics["classes"].items():
        for name in ("precision", "recall", "f1-score", "support"):
            values[f"{label.lower()}_{name.replace('-score', '')}"] = scores[name]
    return {name: value for name, value in values.items() if value is not None}


def metric_values(metadata, metrics):
    values = evaluation_metric_values(metrics)
    values.update(
        {
            key: metadata[key]
            for key in ("training_seconds", "train_rows", "features", "iterations")
            if metadata[key] is not None
        }
    )
    return values


def log_results(client, run_id, files, metadata, metrics):
    parameters = {}
    for group, settings in metadata["settings"].items():
        for name, value in settings.items():
            if isinstance(value, (list, tuple)):
                value = json.dumps(value)
            parameters[f"{group}.{name}"] = value
    parameters.update(
        {
            "input_mode": metadata["input_mode"],
            "labels": json.dumps(metadata["labels"]),
            "dataset_repository": metadata["dataset"]["source"]["repository"],
            "dataset_revision": metadata["dataset"]["source"]["revision"],
            "split_seed": metadata["dataset"]["protocol"]["split"]["seed"],
            "context_version": metadata["dataset"]["protocol"]["context"]["version"],
            "grouping_version": metadata["dataset"]["protocol"]["grouping"]["version"],
            "analyzer": "raw_character_ngrams",
            "tfidf.dtype": "float32",
            "python": sys.version.split()[0],
            "mlflow": version("mlflow"),
            **metadata["versions"],
        }
    )
    timestamp = int(time() * 1000)
    client.log_batch(
        run_id,
        params=[Param(name, str(value)) for name, value in parameters.items()],
        metrics=[
            Metric(name, float(value), timestamp, 0)
            for name, value in metric_values(metadata, metrics).items()
        ],
        synchronous=True,
    )
    for path in files:
        client.log_artifact(run_id, str(path))
    matrix = metrics["confusion_matrix"]
    normalized = dict(
        matrix, counts=[[count / (sum(row) or 1) for count in row] for row in matrix["counts"]]
    )
    client.log_dict(run_id, metrics["classes"], "classification_report.json")
    client.log_dict(run_id, matrix, "confusion_matrix.json")
    client.log_dict(run_id, normalized, "confusion_matrix_normalized.json")


def mark_failed(client, run_id):
    try:
        client.set_terminated(run_id, status="FAILED")
    except (MlflowException, OSError):
        return False
    return True
