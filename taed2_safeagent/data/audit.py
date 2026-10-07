from collections import Counter, defaultdict

from taed2_safeagent.data.common import (
    CONTEXT_FIELDS,
    LABELS,
    read_jsonl,
    write_json,
    write_jsonl,
)
from taed2_safeagent.data.inputs import example_id, safe_context
from taed2_safeagent.data.source import verify_snapshot

RAW_FIELDS = {"command", "session_context", "label", "category", "shell", "reason"}
SHELLS = {"posix", "powershell", "cmd"}


def validate_raw(row):
    if set(row) != RAW_FIELDS:
        raise ValueError("Unexpected raw fields")
    for field in RAW_FIELDS - {"session_context"}:
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"Expected a nonempty string: {field}")
    if row["label"].upper() not in LABELS or row["shell"] not in SHELLS:
        raise ValueError("Unknown label or shell")
    context = row["session_context"]
    if not isinstance(context, dict) or set(context) - set(CONTEXT_FIELDS) - {"assistantMessage"}:
        raise ValueError("Unexpected context fields")
    if "assistantMessage" in context and not isinstance(context["assistantMessage"], str):
        raise ValueError("Invalid assistant message")
    return safe_context(context)


def project_rows(records):
    by_input = defaultdict(list)
    for source, line, row in records:
        context = validate_raw(row)
        identifier = example_id(row["command"], context)
        by_input[identifier].append((source, line, row, context))
    clean, provenance, quarantine = [], [], []
    duplicate_rows = 0
    for identifier, copies in sorted(by_input.items()):
        copies.sort(key=lambda item: (item[0], item[1]))
        labels = {row["label"].upper() for _, _, row, _ in copies}
        conflict = len(labels) > 1
        refs = []
        for source, line, row, _ in copies:
            refs.append(
                {
                    "source_split": source,
                    "line": line,
                    "shell": row["shell"],
                    "category": row["category"],
                    "reason": row["reason"],
                }
            )
            if conflict:
                quarantine.append(
                    {
                        "id": identifier,
                        "source_split": source,
                        "line": line,
                        "row": row,
                        "issue": "conflicting_safe_input_labels",
                    }
                )
        provenance.append(
            {"id": identifier, "status": "quarantined" if conflict else "kept", "sources": refs}
        )
        if not conflict:
            _, _, row, context = copies[0]
            clean.append(
                {
                    "id": identifier,
                    "command": row["command"],
                    "context": context,
                    "label": row["label"].upper(),
                }
            )
            duplicate_rows += len(copies) - 1
    return clean, provenance, quarantine, duplicate_rows


def raw_diagnostics(records):
    commands = defaultdict(set)
    label_counts, shell_counts, missing = Counter(), Counter(), Counter()
    availability = defaultdict(Counter)
    reasons = defaultdict(set)
    reason_rows = Counter()
    for split, _, row in records:
        commands[split].add(row["command"])
        label = row["label"].upper()
        label_counts[label] += 1
        shell_counts[row["shell"]] += 1
        reasons[row["reason"]].add(label)
        reason_rows[row["reason"]] += 1
        for field in (*CONTEXT_FIELDS, "assistantMessage"):
            present = row["session_context"].get(field) is not None
            if not present:
                missing[field] += 1
            availability[f"{field}:{'present' if present else 'missing'}"][label] += 1
    overlap = {}
    for split in ("validation", "test"):
        overlap[split] = sum(
            row["command"] in commands["train"] for name, _, row in records if name == split
        )
    return {
        "labels": label_counts,
        "shells": shell_counts,
        "missing_context_fields": missing,
        "context_availability_by_label": availability,
        "original_command_overlap": overlap,
        "rows_with_class_exclusive_reason": sum(
            reason_rows[reason] for reason, labels in reasons.items() if len(labels) == 1
        ),
    }


def audit(params):
    verify_snapshot(params)
    root = params["paths"]["raw"]
    records = [
        (name.removesuffix(".jsonl"), line, row)
        for name in params["source"]["files"]
        if name.endswith(".jsonl")
        for line, row in read_jsonl(root / name)
    ]
    clean, provenance, quarantine, duplicates = project_rows(records)
    report = {
        "raw_rows": len(records),
        "clean_rows": len(clean),
        "duplicate_rows": duplicates,
        "quarantine_rows": len(quarantine),
        "clean_labels": Counter(row["label"] for row in clean),
        **raw_diagnostics(records),
    }
    for report_key, param_key in (
        ("clean_rows", "expected_clean_rows"),
        ("duplicate_rows", "expected_duplicate_rows"),
        ("quarantine_rows", "expected_quarantine_rows"),
    ):
        if report[report_key] != params["validation"][param_key]:
            raise ValueError(f"Unexpected audit count: {report_key}")
    target = params["paths"]["interim"]
    write_jsonl(target / "clean.jsonl", clean)
    write_jsonl(target / "provenance.jsonl", provenance)
    write_jsonl(target / "quarantine.jsonl", quarantine)
    write_json(params["paths"]["reports"] / "audit.json", report)
    return report
