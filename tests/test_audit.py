from copy import deepcopy

import pytest

from taed2_safeagent.data.audit import project_rows, validate_raw
from taed2_safeagent.data.inputs import build_input, example_id, safe_context


def test_inputs_preserve_command_and_exclude_metadata(raw_row):
    raw_row["session_context"]["assistantMessage"] = "ASSISTANT_HINT"
    context = safe_context(raw_row["session_context"])
    context["reason"] = "REASON_HINT"
    example = {"command": raw_row["command"], "context": context, "label": "LABEL_HINT",
               "category": "CATEGORY_HINT", "shell": "SHELL_HINT", "id": "ID_HINT",
               "group_id": "GROUP_HINT", "source_split": "SOURCE_HINT"}
    text = build_input(example, "command-context")
    assert build_input(example, "command") == raw_row["command"]
    assert raw_row["command"] in text
    assert "HINT" not in text
    assert '"lastUserPrompt":null' in text
    assert '"cwd":null' in text
    with pytest.raises(ValueError, match="mode"):
        build_input(example, "unknown")


def test_path_sets_make_stable_ids(raw_row):
    left = deepcopy(raw_row["session_context"])
    right = deepcopy(left)
    left["gitStatus"]["modified"] = ["B.py", "a.py", "B.py"]
    right["gitStatus"]["modified"] = ["a.py", "B.py"]
    left["agentTouchedFiles"] = ["z", "a", "z"]
    right["agentTouchedFiles"] = ["a", "z"]
    assert example_id(raw_row["command"], safe_context(left)) == example_id(
        raw_row["command"], safe_context(right)
    )
    assert safe_context(left)["gitStatus"]["modified"] == ["B.py", "a.py"]


def test_duplicates_keep_source_mapping(raw_row):
    second = deepcopy(raw_row)
    second["session_context"]["assistantMessage"] = "Different message"
    rows = [("train", 1, raw_row), ("test", 2, second)]
    clean, provenance, quarantine, duplicates = project_rows(rows)
    assert len(clean) == 1
    assert clean[0]["command"] == raw_row["command"]
    assert duplicates == 1
    assert not quarantine
    assert len(provenance[0]["sources"]) == 2
    assert project_rows(list(reversed(rows))) == (clean, provenance, quarantine, duplicates)


def test_conflicting_safe_inputs_are_both_quarantined(raw_row):
    other = deepcopy(raw_row)
    other["label"] = "deny"
    clean, provenance, quarantine, duplicates = project_rows(
        [("train", 1, raw_row), ("validation", 3, other)]
    )
    assert not clean
    assert len(quarantine) == 2
    assert provenance[0]["status"] == "quarantined"
    assert duplicates == 0


def test_same_command_with_different_context_is_kept(raw_row):
    other = deepcopy(raw_row)
    other["label"] = "ask"
    other["session_context"]["lastUserPrompt"] = "Read the status only"
    clean, _, quarantine, duplicates = project_rows(
        [("train", 1, raw_row), ("train", 2, other)]
    )
    assert len(clean) == 2
    assert len({row["id"] for row in clean}) == 2
    assert not quarantine
    assert duplicates == 0


@pytest.mark.parametrize("field,value", [
    ("command", ""), ("command", 42), ("label", "unknown"), ("shell", "unknown"),
    ("reason", None), ("session_context", []),
])
def test_invalid_raw_fields_fail(raw_row, field, value):
    raw_row[field] = value
    with pytest.raises(ValueError):
        validate_raw(raw_row)


@pytest.mark.parametrize("field,value", [
    ("gitStatus", {}), ("gitStatus", None), ("agentTouchedFiles", [4]),
    ("gitRemote", 4), ("lastUserPrompt", []), ("cwd", {}), ("unexpected", "hidden"),
])
def test_invalid_context_fields_fail(raw_row, field, value):
    raw_row["session_context"][field] = value
    with pytest.raises(ValueError):
        validate_raw(raw_row)


def test_required_context_field_cannot_be_missing(raw_row):
    del raw_row["session_context"]["gitRemote"]
    with pytest.raises(ValueError, match="Missing"):
        validate_raw(raw_row)
