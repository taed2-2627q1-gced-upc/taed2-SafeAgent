import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from mlflow import MlflowClient
import pytest
import yaml

from taed2_safeagent.data.common import LABELS, write_json, write_jsonl
from taed2_safeagent.modeling import register
from taed2_safeagent.modeling.common import dataset_identity, load_examples
from taed2_safeagent.modeling.evaluate import compute_metrics


@pytest.fixture(name="cloud_result")
def saved_result(bundle, tmp_path, monkeypatch):
    result = tmp_path / "cloud-result"
    model = result / "model"
    write_json(
        model / "config.json",
        {
            "model_type": "roberta",
            "id2label": {str(index): label for index, label in enumerate(LABELS)},
        },
    )
    (model / "model.safetensors").write_bytes(b"synthetic model for test")
    lock = b"stages: {}\n"
    (result / "input_dvc.lock").write_bytes(lock)
    rows = load_examples(bundle, "validation")
    labels = [row["label"] for row in rows]
    write_json(
        result / "metrics.json",
        {
            "split": "validation",
            "rows": len(rows),
            **compute_metrics(labels, labels),
        },
    )
    write_jsonl(
        result / "predictions.jsonl",
        [
            {"id": row["id"], "true_label": row["label"], "predicted_label": row["label"]}
            for row in rows
        ],
    )
    metadata = {
        "status": "FINISHED",
        "code": {"commit": "a" * 40, "dirty": False},
        "labels": list(LABELS),
        "model": "codebert",
        "input_mode": "command",
        "backbone": bundle["encoders"]["backbones"]["codebert"],
        "dataset": dataset_identity(bundle),
        "settings": bundle["encoders"]["training"],
        "start_time_ms": 1000,
        "end_time_ms": 2000,
        "training_seconds": 1.0,
        "parameters": 10,
        "input_dvc_lock_hash": hashlib.sha256(lock).hexdigest(),
        "model_bytes": sum(path.stat().st_size for path in model.iterdir()),
    }
    write_json(result / "metadata.json", metadata)
    write_json(result / "energy.json", {"status": "unavailable", "country_assumed": True})
    session = tmp_path / "session.json"
    write_json(
        session,
        {
            "git_commit": "a" * 40,
            "kaggle_ref": "joelmrquezalvarez/safeagent-encoder-training",
            "kaggle_version": 2,
            "saved_version_status": "confirmed",
        },
    )
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(json.loads(json.dumps(bundle, default=str))), encoding="utf-8"
    )
    monkeypatch.setattr(register, "code_identity", lambda: {"commit": "b" * 40, "dirty": False})
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="cloud42"))
    client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(status="FINISHED"))
    monkeypatch.setattr(register, "tracking_client", lambda _settings: client)
    commands = []

    def dvc_command(root, *arguments):
        commands.append(arguments)
        if arguments[0] == "add":
            target = root / arguments[1]
            write_json(
                Path(f"{target}.dvc"),
                {
                    "outs": [{"path": target.name, "md5": "c" * 32 + ".dir"}],
                },
            )
        return ""

    monkeypatch.setattr(register, "dvc_command", dvc_command)
    return SimpleNamespace(
        result=result,
        params=bundle,
        params_path=params_path,
        session=session,
        client=client,
        metadata=metadata,
        commands=commands,
        root=tmp_path,
    )


def test_registered_run_keeps_actual_training_source_and_timestamps(cloud_result):
    receipt = register.register_result(
        cloud_result.params_path,
        cloud_result.result,
        cloud_result.session,
    )
    assert receipt["source_commit"] == "a" * 40
    assert receipt["registration_commit"] == "b" * 40
    assert cloud_result.client.create_run.call_args.kwargs["start_time"] == 1000
    cloud_result.client.set_terminated.assert_called_once_with(
        "cloud42",
        end_time=2000,
        status="FINISHED",
    )
    assert [arguments[0] for arguments in cloud_result.commands] == ["add", "push"]
    values = {metric.key for metric in cloud_result.client.log_batch.call_args.kwargs["metrics"]}
    assert "macro_f1" in values
    assert "energy_kwh" not in values
    assert "emissions_kg" not in values
    assert "training_seconds" in values
    assert len(list((cloud_result.root / "reports/encoders").rglob("mlflow_run.json"))) == 1
    assert not list((cloud_result.root / "reports/encoders").rglob("*.jsonl"))
    assert (
        len(list((cloud_result.root / "models/encoders").rglob("validation_predictions.jsonl")))
        == 1
    )


def test_changed_validation_targets_fail_before_shared_run(cloud_result):
    predictions = [
        json.loads(line)
        for line in (cloud_result.result / "predictions.jsonl").read_text().splitlines()
    ]
    predictions[0]["true_label"] = "UNKNOWN"
    write_jsonl(cloud_result.result / "predictions.jsonl", predictions)
    with pytest.raises(ValueError, match="targets differ"):
        register.register_result(
            cloud_result.params_path, cloud_result.result, cloud_result.session
        )
    cloud_result.client.create_run.assert_not_called()


def test_unconfirmed_kaggle_version_is_rejected(cloud_result):
    data = json.loads(cloud_result.session.read_text())
    data["saved_version_status"] = "pending"
    write_json(cloud_result.session, data)
    with pytest.raises(ValueError, match="Confirm the saved"):
        register.register_result(
            cloud_result.params_path, cloud_result.result, cloud_result.session
        )
    cloud_result.client.create_run.assert_not_called()


def test_model_changes_are_rejected_before_upload(cloud_result):
    (cloud_result.result / "model/model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model files differ"):
        register.register_result(
            cloud_result.params_path, cloud_result.result, cloud_result.session
        )
    cloud_result.client.create_run.assert_not_called()


def test_failed_upload_marks_run_failed_without_success_receipt(cloud_result, monkeypatch):
    monkeypatch.setattr(register, "dvc_command", Mock(side_effect=RuntimeError("DVC unavailable")))
    with pytest.raises(RuntimeError, match="inspect the retained"):
        register.register_result(
            cloud_result.params_path, cloud_result.result, cloud_result.session
        )
    cloud_result.client.set_terminated.assert_called_once_with("cloud42", status="FAILED")
    assert not list((cloud_result.root / "reports/encoders").rglob("mlflow_run.json"))


def test_reported_metrics_must_match_saved_predictions(cloud_result):
    metrics = json.loads((cloud_result.result / "metrics.json").read_text())
    metrics["macro_f1"] = 0.4
    write_json(cloud_result.result / "metrics.json", metrics)
    with pytest.raises(ValueError, match="metrics differ"):
        register.register_result(
            cloud_result.params_path, cloud_result.result, cloud_result.session
        )
    cloud_result.client.create_run.assert_not_called()
