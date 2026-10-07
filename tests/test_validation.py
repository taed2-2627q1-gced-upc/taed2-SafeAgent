from copy import deepcopy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from taed2_safeagent import dataset
from taed2_safeagent.data.common import load_params, read_jsonl, write_json, write_jsonl
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.data.split import prepare
from taed2_safeagent.data.validation import deepchecks_checks, great_expectations_checks, validate


def assigned_rows(params):
    return {name: [row for _, row in read_jsonl(params["paths"]["processed"] / f"{name}.jsonl")]
            for name in ("train", "validation", "test")}


def test_fixture_pipeline_and_repeated_outputs(bundle):
    first = validate(bundle)
    assert first["success"]
    assert first["deepchecks"]["sampling"] == "all_rows"
    paths = list(bundle["paths"]["processed"].glob("*.json*"))
    before = {path.name: path.read_bytes() for path in paths}
    prepare(bundle)
    assert {path.name: path.read_bytes() for path in paths} == before
    assert validate(bundle) == first


@pytest.mark.parametrize("field,value", [("label", "DENY"), ("command", "other"),
                                        ("group_id", "wrong"), ("reason", "hidden")])
def test_tampered_prepared_data_fails_and_removes_stale_report(bundle, field, value):
    path = bundle["paths"]["processed"] / "train.jsonl"
    rows = [row for _, row in read_jsonl(path)]
    rows[0][field] = value if rows[0].get(field) != value else "ASK"
    write_jsonl(path, rows)
    report = bundle["paths"]["reports"] / "validation.json"
    write_json(report, {"success": True})
    with pytest.raises(ValueError):
        validate(bundle)
    assert not report.exists()


def test_overlap_and_missing_examples_fail(bundle):
    assigned = assigned_rows(bundle)
    assigned["validation"][0] = deepcopy(assigned["train"][0])
    write_jsonl(bundle["paths"]["processed"] / "validation.jsonl", assigned["validation"])
    with pytest.raises(ValueError, match="cover"):
        validate(bundle)


def test_changed_manifest_fails(bundle):
    path = bundle["paths"]["processed"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["protocol"]["grouping"]["jaccard_numerator"] = 8
    write_json(path, manifest)
    with pytest.raises(ValueError, match="protocol"):
        validate(bundle)


def test_great_expectations_failure_is_an_error(bundle):
    assigned = assigned_rows(bundle)
    assigned["train"][0]["label"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Great Expectations"):
        great_expectations_checks(assigned)


def test_deepchecks_overlap_is_an_error(bundle):
    assigned = assigned_rows(bundle)
    assigned["validation"].append(deepcopy(assigned["train"][0]))
    with pytest.raises(ValueError, match="shared inputs"):
        deepchecks_checks(assigned)


def test_deepchecks_conflicting_full_inputs_are_an_error(bundle):
    assigned = assigned_rows(bundle)
    other = deepcopy(assigned["train"][0])
    other["label"] = "ASK" if other["label"] != "ASK" else "DENY"
    assigned["validation"].append(other)
    with pytest.raises(ValueError, match="conflicting"):
        deepchecks_checks(assigned)


def test_cli_has_help_and_propagates_failed_checks(monkeypatch):
    runner = CliRunner()
    assert runner.invoke(dataset.app, ["--help"]).exit_code == 0
    assert runner.invoke(dataset.app, ["validate", "--help"]).exit_code == 0

    def failed(_params):
        raise ValueError("Expected failure")

    monkeypatch.setattr(dataset, "run_validate", failed)
    result = runner.invoke(dataset.app, ["validate"])
    assert result.exit_code == 1
    assert "Expected failure" in result.output
    assert "Data validation passed" not in result.output


@pytest.mark.integration
def test_full_dataset_contract_and_paired_inputs():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/processed/shell_safety/test.jsonl").is_file():
        pytest.skip("Fetch the pinned snapshot and run dvc repro first")
    params = load_params(root / "params.yaml")
    result = validate(params)
    assert result["success"]
    assert [result["splits"][name]["rows"] for name in ("train", "validation", "test")] == [
        26660, 3333, 3332,
    ]
    for rows in assigned_rows(params).values():
        assert all(build_input(row, "command") == row["command"] for row in rows)
        assert all("CONTEXT\n" in build_input(row, "command-context") for row in rows)
