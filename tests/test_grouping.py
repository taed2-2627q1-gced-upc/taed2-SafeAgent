from copy import deepcopy
from itertools import combinations, product

import pytest

from taed2_safeagent.data.common import LABELS, digest
from taed2_safeagent.data.grouping import (
    build_groups, character_grams, command_fingerprint, near_pairs,
)
from taed2_safeagent.data.inputs import example_id, safe_context
from taed2_safeagent.data.split import assign_groups


def prepared(command, context, label="ALLOW"):
    context = safe_context(context)
    return {"id": example_id(command, context), "command": command,
            "context": context, "label": label}


@pytest.mark.parametrize("command", [
    "git  log  --max-count=123 # wip", "GIT log --max-count=4 & rem fmt",
    "git log --max-count=42 # quick # smoke",
])
def test_synthetic_variants_share_a_fingerprint(params, command):
    assert command_fingerprint(command, params["grouping"]) == "git log --max-count=<num>"


def test_real_comments_and_command_operators_are_kept(params):
    command = "echo ok && rm data # user note"
    assert command_fingerprint(command, params["grouping"]) == command


@pytest.mark.parametrize("size,numerator,denominator", [(1, 3, 4), (5, 9, 10), (3, 1, 1)])
def test_prefix_index_matches_brute_force(params, size, numerator, denominator):
    settings = {**params["grouping"], "ngram_size": size,
                "jaccard_numerator": numerator, "jaccard_denominator": denominator}
    texts = {"".join(chars) for length in range(1, 6) for chars in product("ab", repeat=length)}
    texts.update({"git push remote some_branch_" + suffix for suffix in ("a", "b", "c", "ab")})
    expected = set()
    for left, right in combinations(sorted(texts), 2):
        first, second = character_grams(left, size), character_grams(right, size)
        if denominator * len(first & second) >= numerator * len(first | second):
            expected.add((left, right))
    actual = {tuple(sorted(pair)) for pair in near_pairs(texts, settings)}
    assert actual == expected


def test_neutral_context_does_not_link_unrelated_commands(params, raw_row):
    context = raw_row["session_context"]
    context["gitRemote"] = "git@example.com:team/project.git"
    rows = [prepared(command, context) for command in ("pwd", "echo apples", "sleep infinity")]
    groups, _ = build_groups(rows, params["grouping"])
    assert len(groups) == 3


def test_transitive_groups_and_ids_are_order_independent(params, raw_row):
    left = deepcopy(raw_row["session_context"])
    right = deepcopy(left)
    left["cwd"], right["cwd"] = "/left", "/right"
    rows = [prepared("git log -n 12", left), prepared("git log -n 99", right),
            prepared("pwd", right, "DENY")]
    groups, _ = build_groups(rows, params["grouping"])
    assert len(groups) == 1
    assert build_groups(list(reversed(rows)), params["grouping"])[0] == groups
    rows[0]["label"] = "ASK"
    assert build_groups(rows, params["grouping"])[0] == groups


def test_group_assignment_is_deterministic_and_keeps_whole_groups(params, raw_row):
    context = raw_row["session_context"]
    rows = [prepared(f"command {digest(str(index))}", context, LABELS[index % 3])
            for index in range(90)]
    groups = {digest(row["id"]): [row["id"]] for row in rows}
    assigned = assign_groups(rows, groups, params["split"])
    assert assigned == assign_groups(list(reversed(rows)), dict(reversed(list(groups.items()))),
                                     params["split"])
    assert [len(assigned[name]) for name in ("train", "validation", "test")] == [72, 9, 9]
    seen = set()
    for examples in assigned.values():
        assert {row["label"] for row in examples} == set(LABELS)
        ids = {row["id"] for row in examples}
        assert not seen & ids
        seen |= ids


def test_missing_class_is_an_error(params, raw_row):
    rows = [prepared("pwd", raw_row["session_context"])]
    with pytest.raises(ValueError, match="class"):
        assign_groups(rows, {"group": [rows[0]["id"]]}, params["split"])
