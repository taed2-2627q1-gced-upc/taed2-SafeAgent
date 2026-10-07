from contextlib import contextmanager
import hashlib
from io import BytesIO
import json

import pytest

from taed2_safeagent.data.common import read_jsonl
from taed2_safeagent.data.source import check_file, fetch, verify_snapshot


def source_fixture(params, raw_row):
    payload = (json.dumps(raw_row) + "\n").encode()
    params["source"]["files"] = {"train.jsonl": {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "rows": 1,
    }}
    return payload


def opener_for(payload):
    @contextmanager
    def opener(_url):
        yield BytesIO(payload)
    return opener


def test_fetch_reuses_checked_snapshot(params, raw_row):
    payload = source_fixture(params, raw_row)
    fetch(params, opener_for(payload))
    original = (params["paths"]["raw"] / "train.jsonl").read_bytes()

    def forbidden(_url):
        raise AssertionError("An existing snapshot must not download")

    fetch(params, forbidden)
    verify_snapshot(params)
    assert original == payload


@pytest.mark.parametrize("corruption", ["size", "hash", "rows"])
def test_corrupt_download_is_not_installed(params, raw_row, corruption):
    payload = source_fixture(params, raw_row)
    entry = params["source"]["files"]["train.jsonl"]
    if corruption == "size":
        payload += b" "
    elif corruption == "hash":
        entry["sha256"] = "0" * 64
    else:
        entry["rows"] = 2
    with pytest.raises(ValueError):
        fetch(params, opener_for(payload))
    assert not params["paths"]["raw"].exists()
    assert not list(params["paths"]["raw"].parent.glob(".snapshot_*"))


def test_existing_corrupt_snapshot_is_never_overwritten(params, raw_row):
    payload = source_fixture(params, raw_row)
    fetch(params, opener_for(payload))
    path = params["paths"]["raw"] / "train.jsonl"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="size"):
        fetch(params, opener_for(payload))
    assert path.read_bytes() == b"broken"


def test_interrupted_download_does_not_install_partial_snapshot(params, raw_row):
    payload = source_fixture(params, raw_row)
    params["source"]["files"]["README.md"] = {"bytes": 1, "sha256": "0" * 64}

    @contextmanager
    def interrupted(url):
        if url.endswith("README.md"):
            raise TimeoutError("Interrupted")
        yield BytesIO(payload)

    with pytest.raises(TimeoutError):
        fetch(params, interrupted)
    assert not params["paths"]["raw"].exists()
    assert not list(params["paths"]["raw"].parent.glob(".snapshot_*"))


def test_bad_json_reports_source_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        list(read_jsonl(path))


def test_jsonl_requires_objects(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('[]\n', encoding="utf-8")
    with pytest.raises(TypeError, match="object"):
        list(read_jsonl(path))


def test_check_file_rejects_invalid_json(tmp_path):
    payload = b"not json\n"
    path = tmp_path / "bad.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="Invalid JSON"):
        check_file(path, {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                          "rows": 1})
