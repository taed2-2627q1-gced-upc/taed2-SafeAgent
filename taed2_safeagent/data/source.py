from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

from taed2_safeagent.data.common import read_jsonl, write_json


@contextmanager
def download(url):
    try:
        with urlopen(url, timeout=90) as stream:
            yield stream
    except (URLError, TimeoutError) as error:
        raise ValueError("Could not download the pinned source") from error


def check_file(path, expected):
    payload = Path(path).read_bytes()
    if len(payload) != expected["bytes"]:
        raise ValueError(f"Unexpected size: {Path(path).name}")
    if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError(f"Unexpected hash: {Path(path).name}")
    if "rows" in expected:
        count = sum(1 for _ in read_jsonl(path))
        if count != expected["rows"]:
            raise ValueError(f"Unexpected row count: {Path(path).name}")


def source_manifest(source):
    return {
        "repository": source["repository"],
        "revision": source["revision"],
        "files": source["files"],
    }


def verify_snapshot(params):
    root = params["paths"]["raw"]
    expected = source_manifest(params["source"])
    for name, settings in expected["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid source filename")
        check_file(root / name, settings)
    actual = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("Unexpected source manifest")
    if {path.name for path in root.iterdir()} != set(expected["files"]) | {"manifest.json"}:
        raise ValueError("Unexpected files in the raw snapshot")


def fetch(params, opener=download):
    root = params["paths"]["raw"]
    source = params["source"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", source["repository"]):
        raise ValueError("Invalid dataset name")
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("Expected a pinned source revision")
    if root.exists():
        verify_snapshot(params)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot_", dir=root.parent) as directory:
        staging = Path(directory) / "snapshot"
        staging.mkdir()
        for name, expected in source["files"].items():
            if Path(name).name != name:
                raise ValueError("Invalid source filename")
            url = (
                f"https://huggingface.co/datasets/{source['repository']}/resolve/"
                f"{source['revision']}/{name}"
            )
            with opener(url) as stream, (staging / name).open("wb") as target:
                shutil.copyfileobj(stream, target)
            check_file(staging / name, expected)
        write_json(staging / "manifest.json", source_manifest(source))
        staging.rename(root)
