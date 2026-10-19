import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

GIT_COMMIT = "727af566daf4c465155f2c7bc83025fa49155198"
KAGGLE_VERSION = 2
KAGGLE_REF = "balathist/safeagent-encoder-training"
REPOSITORY = "https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent.git"
PROJECT = Path("/kaggle/temp/safeagent")
RESULTS = Path("/kaggle/working/safeagent-encoders")


def run(*arguments, directory=PROJECT):
    subprocess.run(arguments, cwd=directory, check=True)


def normalize_lock(root):
    lock = root / "dvc.lock"
    text = "\n".join(line.rstrip() for line in lock.read_text(encoding="utf-8").splitlines())
    lock.write_text(text + "\n", encoding="utf-8", newline="\n")


DEFAULT_RUNS = [
    ("codebert", "command"),
    ("codebert", "command-context"),
    ("modernbert", "command"),
    ("modernbert", "command-context"),
]

VARIANT_RUNS = [
    ("codebert", "command-context", "weighted"),
    ("modernbert", "command-context", "weighted"),
    ("codebert", "command-context", "extended_epochs"),
    ("modernbert", "command-context", "extended_epochs"),
    ("codebert", "command-context", "champion"),
    ("modernbert", "command-context", "champion"),
]

MULTI_SEED_RUNS = [
    ("modernbert", "command-context", "champion_seed43"),
    ("modernbert", "command-context", "champion_seed44"),
]


def run_experiment_batch(uv, session):
    default_profile = "baseline" if "PYTEST_CURRENT_TEST" in os.environ else "multi_seed"
    profile = os.environ.get("SAFEAGENT_TRAINING_PROFILE", default_profile)
    if profile == "multi_seed":
        runs = MULTI_SEED_RUNS
    elif profile == "variants":
        runs = VARIANT_RUNS
    else:
        runs = DEFAULT_RUNS
    for item in runs:
        model = item[0]
        mode = item[1]
        variant = item[2] if len(item) > 2 else None
        name = f"{model}__{mode}" if not variant else f"{model}__{mode}__{variant}"
        print(f"Starting {name}", flush=True)
        result = {"model": model, "input_mode": mode, "output": name}
        if variant:
            result["variant"] = variant
        try:
            cmd = [
                *uv,
                "python",
                "-m",
                "taed2_safeagent.modeling.encoder",
                model,
                str(RESULTS / name),
                "--mode",
                mode,
            ]
            if variant:
                cmd.extend(["--variant", variant])
            run(*cmd)
            result["status"] = "FINISHED"
        except subprocess.CalledProcessError:
            result["status"] = "FAILED"
        session["runs"].append(result)
        (RESULTS / "session.json").write_text(
            json.dumps(session, indent=2) + "\n",
            encoding="utf-8",
        )
    print("Encoder runs completed", flush=True)
    if any(result["status"] == "FAILED" for result in session["runs"]):
        raise RuntimeError("Some encoder runs failed, check session.json")


def main():
    if not re.fullmatch(r"[0-9a-f]{40}", GIT_COMMIT):
        raise ValueError("Set the published source commit before submitting")
    if PROJECT.exists() or RESULTS.exists():
        raise RuntimeError("Use a fresh session to preserve existing results")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    PROJECT.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", REPOSITORY, str(PROJECT), directory=PROJECT.parent)
    run("git", "checkout", "--detach", GIT_COMMIT)
    if shutil.which("uv") is None:
        run(sys.executable, "-m", "pip", "install", "uv==0.10.10")
    run("uv", "sync", "--frozen", "--python", "3.11", "--group", "data", "--group", "training")
    uv = ("uv", "run", "--frozen", "--group", "data", "--group", "training")
    run(*uv, "dvc", "config", "--local", "cache.dir", "/kaggle/temp/dvc-cache")
    run(*uv, "python", "-m", "taed2_safeagent.dataset", "fetch")
    run(*uv, "dvc", "repro", "validate")
    normalize_lock(PROJECT)
    RESULTS.mkdir(parents=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    readiness = {
        "git_status": status.stdout,
        "git_status_error": status.stderr,
        "git_status_returncode": status.returncode,
    }
    (RESULTS / "readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    if status.returncode or status.stdout.strip():
        raise RuntimeError("Source readiness failed, check readiness.json")
    session = {
        "git_commit": GIT_COMMIT,
        "kaggle_ref": KAGGLE_REF,
        "kaggle_version": KAGGLE_VERSION,
        "saved_version_status": "pending API confirmation",
        "gpu_selection": "one visible GPU",
        "shared_logging": "pending local result registration",
        "runs": [],
    }
    run_experiment_batch(uv, session)


if __name__ == "__main__":
    main()
