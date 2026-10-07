import hashlib
import json
from pathlib import Path

import yaml

LABELS = ("ALLOW", "ASK", "DENY")
SPLITS = ("train", "validation", "test")
CONTEXT_FIELDS = ("gitRemote", "gitStatus", "agentTouchedFiles", "lastUserPrompt", "cwd")
STATUS_FIELDS = ("untracked", "modified", "staged")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {Path(path).name}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {Path(path).name}:{line_number}")
            yield line_number, row


def load_params(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    if params["context"]["version"] != "safe_context_v1":
        raise ValueError("Unknown context profile")
    if params["grouping"]["version"] != "command_context_v1":
        raise ValueError("Unknown grouping profile")
    ratios = params["split"]["ratios"]
    if set(ratios) != set(SPLITS) or any(value <= 0 for value in ratios.values()):
        raise ValueError("Expected three positive split ratios")
    if abs(sum(ratios.values()) - 1) > 1e-12:
        raise ValueError("Split ratios must sum to one")
    group = params["grouping"]
    if not 0 < group["jaccard_numerator"] <= group["jaccard_denominator"]:
        raise ValueError("Invalid similarity threshold")
    if group["ngram_size"] < 1:
        raise ValueError("Invalid gram size")
    for name, relative in params["paths"].items():
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(path.parent) or resolved == path.parent:
            raise ValueError(f"Invalid data path: {name}")
        params["paths"][name] = resolved
    return params
