import json
import subprocess

import pytest

import kaggle_train


def test_cloud_script_rejects_missing_source_before_commands(monkeypatch):
    execute = []
    monkeypatch.setattr(kaggle_train, "GIT_COMMIT", "")
    monkeypatch.setattr(kaggle_train, "run", lambda *_args, **_kwargs: execute.append(True))
    with pytest.raises(ValueError, match="published source"):
        kaggle_train.main()
    assert not execute


def test_cloud_script_preserves_other_runs_after_failure(tmp_path, monkeypatch):
    project = tmp_path / "project"
    results = tmp_path / "results"
    commands = []
    monkeypatch.setattr(kaggle_train, "GIT_COMMIT", "a" * 40)
    monkeypatch.setattr(kaggle_train, "PROJECT", project)
    monkeypatch.setattr(kaggle_train, "RESULTS", results)
    monkeypatch.setattr(kaggle_train.shutil, "which", lambda _command: "/uv")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "")

    def execute(*arguments, directory=project):
        commands.append(arguments)
        assert directory in {project, project.parent}
        if arguments[:2] == ("git", "clone"):
            project.mkdir()
        if (
            "taed2_safeagent.modeling.encoder" in arguments
            and arguments[-1] == "command"
            and "codebert" in arguments
        ):
            raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(kaggle_train, "run", execute)
    with pytest.raises(RuntimeError, match="Some encoder runs failed"):
        kaggle_train.main()
    session = json.loads((results / "session.json").read_text(encoding="utf-8"))
    assert len(session["runs"]) == 4
    assert [run["status"] for run in session["runs"]] == [
        "FAILED",
        "FINISHED",
        "FINISHED",
        "FINISHED",
    ]
    assert kaggle_train.os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert (
        "uv",
        "run",
        "--frozen",
        "--group",
        "data",
        "--group",
        "training",
        "dvc",
        "repro",
        "validate",
    ) in commands
