from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest
import yaml

from taed2_safeagent.data.common import write_json
from taed2_safeagent.data.validation import validate
from taed2_safeagent.modeling.train import train


def run_dvc(root, *arguments):
    command = [sys.executable, "-m", "dvc", *arguments]
    return subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


@pytest.fixture(name="pipeline_project")
def small_pipeline(baseline_params, tmp_path):
    source = Path(__file__).resolve().parents[1]
    settings = deepcopy(baseline_params)
    validate(settings)
    write_json(
        tmp_path / "reports/data/validation.json",
        json.loads((settings["paths"]["reports"] / "validation.json").read_text(encoding="utf-8")),
    )
    settings["paths"] = {name: str(path) for name, path in settings["paths"].items()}
    settings["baseline"]["model_dir"] = "models/baseline"
    settings["baseline"]["report_dir"] = "reports/baseline"
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    shutil.copytree(
        source / "taed2_safeagent",
        tmp_path / "taed2_safeagent",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(source / name, tmp_path / name)
    stages = yaml.safe_load((source / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    stages = {name: stages[name] for name in ("train", "evaluate")}
    for stage in stages.values():
        stage["cmd"] = stage["cmd"].replace(
            "uv run --frozen --group data python", f'"{sys.executable}"'
        )
    (tmp_path / "dvc.yaml").write_text(yaml.safe_dump({"stages": stages}), encoding="utf-8")
    result = run_dvc(tmp_path, "init", "--no-scm")
    assert result.returncode == 0, result.stderr
    with TemporaryDirectory(prefix="sa-cache-") as cache:
        result = run_dvc(tmp_path, "config", "cache.dir", cache)
        assert result.returncode == 0, result.stderr
        yield tmp_path, settings


def test_dvc_reuses_outputs_and_retrains_after_settings_change(pipeline_project):
    root, settings = pipeline_project
    result = run_dvc(root, "repro")
    assert result.returncode == 0, result.stderr
    model = root / "models/baseline/metadata.json"
    first = model.read_bytes()
    result = run_dvc(root, "repro")
    assert result.returncode == 0, result.stderr
    assert model.read_bytes() == first
    assert json.loads(run_dvc(root, "status", "--json").stdout) == {}
    settings["baseline"]["svm"]["C"] = 0.5
    (root / "params.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    changed = json.loads(run_dvc(root, "status", "--json").stdout)
    assert "train" in changed
    result = run_dvc(root, "repro")
    assert result.returncode == 0, result.stderr
    assert json.loads(model.read_text(encoding="utf-8"))["settings"]["svm"]["C"] == 0.5
    metrics = json.loads((root / "reports/baseline/metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"] == "validation"
    assert metrics["settings"]["svm"]["C"] == 0.5
    assert not (root / "reports/baseline/mlflow_run.json").exists()
    lock = yaml.safe_load((root / "dvc.lock").read_text(encoding="utf-8"))
    assert lock["stages"]["train"]["outs"][0]["path"] == "models/baseline"


def test_failed_training_stops_pipeline_before_evaluation(pipeline_project):
    root, settings = pipeline_project
    settings["baseline"]["tfidf"]["min_df"] = 10000
    (root / "params.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    result = run_dvc(root, "repro")
    assert result.returncode != 0
    assert "Running stage 'evaluate'" not in result.stdout
    assert not (root / "reports/baseline/metrics.json").exists()
    assert not (root / "reports/baseline/mlflow_run.json").exists()


def test_missing_quality_report_blocks_model_pipeline(pipeline_project):
    root, _ = pipeline_project
    (root / "reports/data/validation.json").unlink()
    result = run_dvc(root, "repro")
    assert result.returncode != 0
    assert not (root / "models/baseline").exists()


def test_new_training_clears_previous_tracking_receipt(baseline_params):
    receipt = baseline_params["baseline"]["report_dir"] / "mlflow_run.json"
    write_json(receipt, {"status": "FINISHED", "run_id": "previous"})
    train(baseline_params)
    assert not receipt.exists()
