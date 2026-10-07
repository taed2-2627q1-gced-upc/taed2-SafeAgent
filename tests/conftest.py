from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from taed2_safeagent.data.audit import audit
from taed2_safeagent.data.common import (
    LABELS,
    canonical_json,
    digest,
    load_params,
    write_json,
    write_jsonl,
)
from taed2_safeagent.data.source import source_manifest
from taed2_safeagent.data.split import prepare


@pytest.fixture(name="params")
def dataset_params(tmp_path):
    settings = load_params(Path(__file__).resolve().parents[1] / "params.yaml")
    settings["paths"] = {name: tmp_path / name for name in settings["paths"]}
    return settings


@pytest.fixture(name="raw_row")
def source_row():
    return {"command": "  git status  ", "session_context": {
        "gitRemote": None,
        "gitStatus": {"untracked": [], "modified": [], "staged": []},
        "agentTouchedFiles": [],
    }, "label": "allow", "category": "git", "shell": "posix", "reason": "local_read"}


@pytest.fixture(name="bundle")
def data_bundle(params, raw_row):
    rows = []
    for index in range(180):
        row = deepcopy(raw_row)
        row["command"] = "printf " + digest(str(index))
        row["label"] = LABELS[index % 3].lower()
        row["shell"] = ("posix", "powershell", "cmd")[index % 3]
        rows.append(row)
    payload = ("\n".join(canonical_json(row) for row in rows) + "\n").encode()
    params["source"]["files"] = {"train.jsonl": {
        "rows": len(rows), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }}
    params["validation"] = {"expected_clean_rows": len(rows), "expected_duplicate_rows": 0,
                            "expected_quarantine_rows": 0, "require_ambiguous_commands": False}
    raw = params["paths"]["raw"]
    write_jsonl(raw / "train.jsonl", rows)
    write_json(raw / "manifest.json", source_manifest(params["source"]))
    audit(params)
    prepare(params)
    return params


@pytest.fixture(name="baseline_params")
def baseline_settings(bundle, tmp_path):
    bundle["baseline"]["model_dir"] = tmp_path / "models" / "baseline"
    bundle["baseline"]["report_dir"] = tmp_path / "reports" / "baseline"
    return bundle
