import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.client import TrackingServiceClient
import pytest
from typer.testing import CliRunner

from taed2_safeagent.data.common import write_json
from taed2_safeagent.modeling import experiment, tracking
from taed2_safeagent.modeling import train as training


@pytest.fixture(name="tracked_run")
def tracking_setup(baseline_params, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "jaycoding205")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "test_secret")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    (tmp_path / "dvc.lock").write_text("stages: {}\n", encoding="utf-8")
    source = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(tracking, "code_identity", lambda: source)
    monkeypatch.setattr(training, "code_identity", lambda: source)
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="7")
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run42"))
    client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(status="FINISHED"))
    factory = Mock(return_value=client)
    monkeypatch.setattr(tracking, "MlflowClient", factory)
    commands = []

    def fake_dvc(root, *arguments):
        commands.append(arguments)
        if arguments[0] == "commit":
            write_json(
                root / "dvc.lock",
                {
                    "stages": {
                        "train": {
                            "outs": [
                                {
                                    "md5": "c" * 32 + ".dir",
                                    "path": baseline_params["baseline"]["model_dir"]
                                    .relative_to(root)
                                    .as_posix(),
                                }
                            ]
                        }
                    }
                },
            )
        return "{}" if arguments[0] == "status" else ""

    monkeypatch.setattr(tracking, "dvc_command", fake_dvc)
    return SimpleNamespace(
        params=baseline_params,
        root=tmp_path,
        client=client,
        source=source,
        commands=commands,
        factory=factory,
    )


def test_shared_run_records_real_outputs_and_model_version(tracked_run):
    result = experiment.run_experiment(tracked_run.params, tracked_run.root)
    client = tracked_run.client
    stored = json.loads(
        (tracked_run.params["baseline"]["report_dir"] / "mlflow_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == result
    assert result["status"] == "FINISHED"
    assert result["source_commit"] == tracked_run.source["commit"]
    assert result["model_dvc_version"] == "c" * 32 + ".dir"
    tracked_run.factory.assert_called_once_with(tracking_uri=tracked_run.params["tracking"]["uri"])
    client.create_run.assert_called_once()
    assert client.create_run.call_args.kwargs["run_name"].endswith("__command__seed42")
    assert client.create_run.call_args.kwargs["tags"]["evaluation_split"] == "validation"
    assert tracked_run.commands == [
        ("status", "--json", "audit", "prepare", "validate"),
        ("commit", "--force", "train", "evaluate"),
        ("push", "train"),
    ]
    assert result["model_dvc_path"] == "dvc.lock"
    assert result["model_dvc_stage"] == "train"
    model = tracked_run.params["baseline"]["model_dir"]
    metadata = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (tracked_run.params["baseline"]["report_dir"] / "metrics.json").read_text(encoding="utf-8")
    )
    assert metadata["tracking"]["run_id"] == result["run_id"]
    values = {metric.key: metric.value for metric in client.log_batch.call_args.kwargs["metrics"]}
    assert values["macro_f1"] == metrics["macro_f1"]
    assert values["deny_recall"] == metrics["classes"]["DENY"]["recall"]
    assert values["deny_to_allow_count"] == metrics["deny_to_allow_count"]
    assert values["train_rows"] == metadata["train_rows"]
    settings = {param.key: param.value for param in client.log_batch.call_args.kwargs["params"]}
    assert settings["svm.random_state"] == "42"
    assert settings["tfidf.ngram_range"] == "[2, 5]"
    assert settings["input_mode"] == "command"
    artifacts = {Path(call.args[1]).name for call in client.log_artifact.call_args_list}
    assert artifacts == {
        "input_dvc.lock",
        "metadata.json",
        "dvc.lock",
        "metrics.json",
        "predictions.jsonl",
    }
    assert "test_secret" not in str(settings)
    assert "test_secret" not in str(client.log_dict.call_args_list)
    matrices = {call.args[2]: call.args[1] for call in client.log_dict.call_args_list}
    normalized = matrices["confusion_matrix_normalized.json"]
    assert normalized["labels"] == ["ALLOW", "ASK", "DENY"]
    assert all(sum(row) == pytest.approx(1) for row in normalized["counts"])
    client.set_terminated.assert_called_once_with("run42", status="FINISHED")


@pytest.mark.parametrize("content", [b"stages: {}\n", b"stages: {}\r\n"])
def test_input_lock_artifact_preserves_recorded_bytes(content):
    client = Mock(spec=MlflowClient)
    uploaded = []

    def capture(_run_id, path):
        uploaded.append(Path(path).read_bytes())

    client.log_artifact.side_effect = capture
    tracking.log_input_lock(client, "run42", content)
    assert uploaded == [content]


@pytest.mark.parametrize("variable", ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"])
def test_missing_auth_prevents_training(tracked_run, monkeypatch, variable):
    monkeypatch.delenv(variable)
    with pytest.raises(ValueError, match="Missing environment"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    tracked_run.client.create_run.assert_not_called()
    assert not tracked_run.params["baseline"]["model_dir"].exists()


@pytest.mark.parametrize(
    "uri",
    [
        "file:///tmp/mlruns",
        "https://other.example/model.mlflow",
        "https://secret@dagshub.com/team/repo.mlflow",
    ],
)
def test_invalid_destinations_are_rejected(tracked_run, uri):
    tracked_run.params["tracking"]["uri"] = uri
    with pytest.raises(ValueError, match="DagsHub"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    tracked_run.factory.assert_not_called()


def test_conflicting_uri_is_rejected(tracked_run, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/other/repo.mlflow")
    with pytest.raises(ValueError, match="differs"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    tracked_run.client.create_run.assert_not_called()


def test_dirty_source_prevents_run_creation(tracked_run):
    tracked_run.source["dirty"] = True
    with pytest.raises(ValueError, match="Commit"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    tracked_run.client.create_run.assert_not_called()


def test_changed_data_prevents_run_creation(tracked_run, monkeypatch):
    monkeypatch.setattr(tracking, "dvc_command", lambda *_args: '{"prepare": []}')
    with pytest.raises(ValueError, match="DVC inputs"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    tracked_run.client.create_run.assert_not_called()


def test_new_experiment_is_created_only_when_missing(tracked_run):
    tracked_run.client.get_experiment_by_name.return_value = None
    tracked_run.client.create_experiment.return_value = "7"
    experiment.run_experiment(tracked_run.params, tracked_run.root)
    tracked_run.client.create_experiment.assert_called_once_with("SafeAgent")


@pytest.mark.parametrize("stage", ["train", "evaluate", "commit", "push", "log_artifact"])
def test_failed_stages_mark_run_failed_without_success_receipt(tracked_run, monkeypatch, stage):
    error = MlflowException("test_secret")
    if stage in {"train", "evaluate"}:
        monkeypatch.setattr(experiment, stage, Mock(side_effect=error))
    elif stage == "log_artifact":
        tracked_run.client.log_artifact.side_effect = error
    else:
        original = tracking.dvc_command

        def fail_dvc(root, *arguments):
            if arguments[0] == stage:
                raise RuntimeError("DVC failed")
            return original(root, *arguments)

        monkeypatch.setattr(tracking, "dvc_command", fail_dvc)
    with pytest.raises(RuntimeError, match="marked failed") as failure:
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    assert "test_secret" not in str(failure.value)
    tracked_run.client.set_terminated.assert_called_once_with("run42", status="FAILED")
    assert not (tracked_run.params["baseline"]["report_dir"] / "mlflow_run.json").exists()


def test_failed_validation_removes_old_success_receipt(tracked_run, monkeypatch):
    report = tracked_run.params["baseline"]["report_dir"] / "mlflow_run.json"
    write_json(report, {"status": "FINISHED", "run_id": "old_run"})
    monkeypatch.setattr(experiment, "evaluate", Mock(side_effect=ValueError("Bad data")))
    with pytest.raises(RuntimeError):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    assert not report.exists()


def test_unconfirmed_remote_failure_is_reported(tracked_run, monkeypatch):
    monkeypatch.setattr(experiment, "train", Mock(side_effect=ValueError("Bad inputs")))
    tracked_run.client.set_terminated.side_effect = MlflowException("test_secret")
    with pytest.raises(RuntimeError, match="not confirmed"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)


def test_unfinished_remote_run_has_no_success_receipt(tracked_run):
    tracked_run.client.get_run.return_value.info.status = "RUNNING"
    with pytest.raises(RuntimeError, match="run completion"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    assert not (tracked_run.params["baseline"]["report_dir"] / "mlflow_run.json").exists()


def test_cli_redacts_backend_details(tracked_run, monkeypatch):
    monkeypatch.setattr(experiment, "load_config", lambda _path: tracked_run.params)
    tracked_run.client.log_batch.side_effect = MlflowException("test_secret")
    runner = CliRunner()
    assert runner.invoke(experiment.app, ["--help"]).exit_code == 0
    result = runner.invoke(experiment.app, ["--params", str(tracked_run.root / "params.yaml")])
    assert result.exit_code == 1
    assert "result logging" in result.output
    assert "test_secret" not in result.output


def test_unavailable_server_prevents_training(tracked_run):
    tracked_run.client.get_experiment_by_name.side_effect = MlflowException("test_secret")
    with pytest.raises(RuntimeError, match="start the shared experiment"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    assert not tracked_run.params["baseline"]["model_dir"].exists()


def test_changed_source_during_training_is_rejected(tracked_run, monkeypatch):
    monkeypatch.setattr(training, "code_identity", lambda: {"commit": "b" * 40, "dirty": False})
    with pytest.raises(RuntimeError, match="during training"):
        experiment.run_experiment(tracked_run.params, tracked_run.root)
    tracked_run.client.set_terminated.assert_called_once_with("run42", status="FAILED")
    assert [arguments[0] for arguments in tracked_run.commands] == ["status"]


def test_dvc_failure_uses_project_python_and_redacts_output(tmp_path, monkeypatch):
    execute = Mock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="test_secret"))
    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="DVC push failed") as failure:
        tracking.dvc_command(tmp_path, "push", "train")
    assert "test_secret" not in str(failure.value)
    assert execute.call_args.args[0] == [
        sys.executable,
        "-m",
        "dvc",
        "push",
        "train",
    ]
    assert execute.call_args.kwargs["cwd"] == tmp_path


def test_sdk_completion_works_with_plain_text_output(tracked_run, monkeypatch):
    tracking.tracking_client(tracked_run.params["tracking"])
    store = Mock(spec=RestStore)
    monkeypatch.setattr(TrackingServiceClient, "store", property(lambda _self: store))
    client = MlflowClient(tracking_uri=tracked_run.params["tracking"]["uri"])
    with (
        io.TextIOWrapper(io.BytesIO(), encoding="ascii") as output,
        monkeypatch.context() as scoped,
    ):
        scoped.setattr(sys, "stdout", output)
        client.set_terminated("run42", status="FINISHED")
    store.update_run_info.assert_called_once()
    store.get_run.assert_not_called()
