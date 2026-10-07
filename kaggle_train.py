import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

GIT_COMMIT = ""
KAGGLE_VERSION = 1
KAGGLE_REF = "joelmrquezalvarez/safeagent-encoder-training"
REPOSITORY = "https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent.git"
PROJECT = Path("/kaggle/temp/safeagent")
RESULTS = Path("/kaggle/working/safeagent-encoders")


def run(*arguments, directory=PROJECT):
    subprocess.run(arguments, cwd=directory, check=True)


def normalize_lock(root):
    lock = root / "dvc.lock"
    text = "\n".join(line.rstrip() for line in lock.read_text(encoding="utf-8").splitlines())
    lock.write_text(text + "\n", encoding="utf-8", newline="\n")


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
    for model in ("codebert", "modernbert"):
        for mode in ("command", "command-context"):
            name = f"{model}__{mode}"
            print(f"Starting {name}", flush=True)
            result = {"model": model, "input_mode": mode, "output": name}
            try:
                run(
                    *uv,
                    "python",
                    "-m",
                    "taed2_safeagent.modeling.encoder",
                    model,
                    str(RESULTS / name),
                    "--mode",
                    mode,
                )
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


if __name__ == "__main__":
    main()
