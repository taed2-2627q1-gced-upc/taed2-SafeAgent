from pathlib import Path

import pytest

from taed2_safeagent.data.common import load_params


@pytest.fixture
def params(tmp_path):
    settings = load_params(Path(__file__).resolve().parents[1] / "params.yaml")
    settings["paths"] = {name: tmp_path / name for name in settings["paths"]}
    return settings


@pytest.fixture
def raw_row():
    return {"command": "  git status  ", "session_context": {
        "gitRemote": None,
        "gitStatus": {"untracked": [], "modified": [], "staged": []},
        "agentTouchedFiles": [],
    }, "label": "allow", "category": "git", "shell": "posix", "reason": "local_read"}
