from collections import Counter, defaultdict

from taed2_safeagent.data.common import (
    LABELS,
    SPLITS,
    digest,
    read_jsonl,
    write_json,
    write_jsonl,
)
from taed2_safeagent.data.grouping import build_groups


def allocation_score(current, addition, target):
    return ((current + addition - target) ** 2 - (current - target) ** 2) / target


def assign_groups(rows, groups, settings):
    by_id = {row["id"]: row for row in rows}
    total = Counter(row["label"] for row in rows)
    if set(total) != set(LABELS):
        raise ValueError("Every class must be present before splitting")
    sizes = Counter()
    counts = {split: Counter() for split in SPLITS}
    assigned = {split: [] for split in SPLITS}
    order = sorted(
        groups,
        key=lambda group: (
            -len(groups[group]),
            digest(f"{settings['seed']}:{group}"),
            group,
        ),
    )
    for group in order:
        members = groups[group]
        labels = Counter(by_id[identifier]["label"] for identifier in members)
        scores = {}
        for split in SPLITS:
            ratio = settings["ratios"][split]
            score = allocation_score(sizes[split], len(members), len(rows) * ratio)
            score += sum(
                allocation_score(counts[split][label], labels[label], total[label] * ratio)
                for label in LABELS
            )
            scores[split] = score
        chosen = min(SPLITS, key=scores.__getitem__)
        sizes[chosen] += len(members)
        counts[chosen].update(labels)
        assigned[chosen].extend({**by_id[identifier], "group_id": group} for identifier in members)
    for examples in assigned.values():
        examples.sort(key=lambda row: row["id"])
    return assigned


def split_diagnostics(assigned, provenance):
    sources = {row["id"]: row["sources"] for row in provenance if row["status"] == "kept"}
    report = {}
    for split, rows in assigned.items():
        command_labels = defaultdict(set)
        shells = Counter()
        for row in rows:
            command_labels[row["command"]].add(row["label"])
            shell = sources[row["id"]][0]["shell"]
            shells[shell] += 1
        report[split] = {
            "rows": len(rows),
            "labels": Counter(row["label"] for row in rows),
            "shells": shells,
            "groups": len({row["group_id"] for row in rows}),
            "ambiguous_commands": sum(len(labels) > 1 for labels in command_labels.values()),
        }
    return report


def prepare(params):
    interim = params["paths"]["interim"]
    rows = [row for _, row in read_jsonl(interim / "clean.jsonl")]
    provenance = [row for _, row in read_jsonl(interim / "provenance.jsonl")]
    groups, grouping = build_groups(rows, params["grouping"])
    assigned = assign_groups(rows, groups, params["split"])
    report = {"grouping": grouping, "splits": split_diagnostics(assigned, provenance)}
    processed = params["paths"]["processed"]
    for split, examples in assigned.items():
        write_jsonl(processed / f"{split}.jsonl", examples)
    write_json(
        processed / "manifest.json",
        {
            "source": {key: params["source"][key] for key in ("repository", "revision")},
            "protocol": {key: params[key] for key in ("context", "grouping", "split")},
            "labels": list(LABELS),
            "input_modes": ["command", "command-context"],
            **report,
        },
    )
    write_json(params["paths"]["reports"] / "split.json", report)
    return report
