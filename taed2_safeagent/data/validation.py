from collections import Counter
from importlib.metadata import version
from itertools import combinations
import json

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ConflictingLabels, TrainTestSamplesMix
import great_expectations as gx
import pandas as pd

from taed2_safeagent.data.audit import project_rows
from taed2_safeagent.data.common import LABELS, SPLITS, read_jsonl, write_json
from taed2_safeagent.data.grouping import build_groups
from taed2_safeagent.data.inputs import build_input
from taed2_safeagent.data.source import verify_snapshot
from taed2_safeagent.data.split import assign_groups, split_diagnostics

PREPARED_FIELDS = {"id", "group_id", "command", "context", "label"}


def validate_structure(assigned, clean, provenance, params):
    expected = {row["id"]: row for row in clean}
    all_rows = []
    for rows in assigned.values():
        for row in rows:
            if set(row) != PREPARED_FIELDS:
                raise ValueError("Unexpected prepared fields")
            original = {key: value for key, value in row.items() if key != "group_id"}
            if original != expected.get(row["id"]):
                raise ValueError("Prepared input differs from audited input")
            all_rows.append(row)
    if len(all_rows) != len(expected) or {row["id"] for row in all_rows} != set(expected):
        raise ValueError("Prepared inputs must cover each audited example once")
    groups, grouping = build_groups(clean, params["grouping"])
    membership = {identifier: group for group, members in groups.items() for identifier in members}
    if any(row["group_id"] != membership[row["id"]] for row in all_rows):
        raise ValueError("Invalid group membership")
    for left, right in combinations(SPLITS, 2):
        for field in ("id", "group_id"):
            if {row[field] for row in assigned[left]} & {row[field] for row in assigned[right]}:
                raise ValueError(f"Split overlap: {field}")
    if assigned != assign_groups(clean, groups, params["split"]):
        raise ValueError("Split assignment differs from the frozen protocol")
    report = {"grouping": grouping, "splits": split_diagnostics(assigned, provenance)}
    validate_distribution(report, clean, params)
    return report


def validate_distribution(report, clean, params):
    totals = Counter(row["label"] for row in clean)
    tolerance = params["split"]["tolerance"]
    for split, stats in report["splits"].items():
        ratio = params["split"]["ratios"][split]
        if abs(stats["rows"] / len(clean) - ratio) > tolerance:
            raise ValueError("Split size differs from the target")
        if set(stats["labels"]) != set(LABELS):
            raise ValueError("A split is missing a class")
        for label in LABELS:
            if abs(stats["labels"][label] / totals[label] - ratio) > tolerance:
                raise ValueError("Class size differs from the target")
        if set(stats["shells"]) != {"posix", "powershell", "cmd"}:
            raise ValueError("A split is missing a shell")
        if (
            split != "train"
            and params["validation"]["require_ambiguous_commands"]
            and stats["ambiguous_commands"] == 0
        ):
            raise ValueError("Evaluation must include ambiguous commands")


def great_expectations_checks(assigned):
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="safeagent")
    results = {}
    for split, rows in assigned.items():
        frame = pd.DataFrame([{**row, "context": json.dumps(row["context"])} for row in rows])
        asset = source.add_dataframe_asset(name=split)
        batch = asset.add_batch_definition_whole_dataframe(name="all_rows").get_batch(
            batch_parameters={"dataframe": frame}
        )
        expectations = [
            gx.expectations.ExpectTableColumnsToMatchSet(
                column_set=sorted(PREPARED_FIELDS), exact_match=True
            ),
            gx.expectations.ExpectTableRowCountToEqual(value=len(rows)),
            gx.expectations.ExpectColumnValuesToBeInSet(column="label", value_set=list(LABELS)),
            gx.expectations.ExpectColumnValuesToBeUnique(column="id"),
            gx.expectations.ExpectColumnValueLengthsToBeBetween(column="command", min_value=1),
        ]
        expectations.extend(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
            for column in sorted(PREPARED_FIELDS)
        )
        passed = [bool(batch.validate(expectation).success) for expectation in expectations]
        if not all(passed):
            raise ValueError(f"Great Expectations failed: {split}")
        results[split] = {"rows": len(rows), "expectations": len(passed), "success": True}
    return {"version": version("great-expectations"), "results": results}


def deepchecks_checks(assigned):
    frames = {
        split: pd.DataFrame(
            {
                "input": [build_input(row, "command-context") for row in rows],
                "label": [row["label"] for row in rows],
            }
        )
        for split, rows in assigned.items()
    }
    full = pd.concat(list(frames.values()), ignore_index=True)

    def dataset(frame):
        return Dataset(frame, label="label", label_type="multiclass", cat_features=["input"])

    result = (
        ConflictingLabels(n_samples=len(full))
        .add_condition_ratio_of_conflicting_labels_less_or_equal(0)
        .run(dataset(full))
    )
    if not result.passed_conditions():
        raise ValueError("Deepchecks found conflicting full inputs")
    results = {"conflicting_labels": {"success": True, "rows": len(full)}}
    for left, right in combinations(SPLITS, 2):
        check = TrainTestSamplesMix(
            n_samples=len(full)
        ).add_condition_duplicates_ratio_less_or_equal(0)
        result = check.run(
            train_dataset=dataset(frames[left]), test_dataset=dataset(frames[right])
        )
        if not result.passed_conditions():
            raise ValueError(f"Deepchecks found shared inputs: {left}, {right}")
        results[f"{left}:{right}"] = {
            "success": True,
            "rows": {left: len(frames[left]), right: len(frames[right])},
        }
    return {"version": version("deepchecks"), "sampling": "all_rows", "results": results}


def validate(params):
    report_path = params["paths"]["reports"] / "validation.json"
    report_path.unlink(missing_ok=True)
    verify_snapshot(params)
    raw = params["paths"]["raw"]
    records = [
        (name.removesuffix(".jsonl"), line, row)
        for name in params["source"]["files"]
        if name.endswith(".jsonl")
        for line, row in read_jsonl(raw / name)
    ]
    clean, provenance, quarantine, _ = project_rows(records)
    interim = params["paths"]["interim"]
    if clean != [row for _, row in read_jsonl(interim / "clean.jsonl")]:
        raise ValueError("Audited inputs differ from the pinned source")
    if provenance != [row for _, row in read_jsonl(interim / "provenance.jsonl")]:
        raise ValueError("Provenance differs from the pinned source")
    if quarantine != [row for _, row in read_jsonl(interim / "quarantine.jsonl")]:
        raise ValueError("Quarantine differs from the pinned source")
    processed = params["paths"]["processed"]
    assigned = {
        split: [row for _, row in read_jsonl(processed / f"{split}.jsonl")] for split in SPLITS
    }
    report = validate_structure(assigned, clean, provenance, params)
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    for key in ("context", "grouping", "split"):
        if manifest["protocol"][key] != params[key]:
            raise ValueError(f"Manifest differs from the protocol: {key}")
    if manifest["source"] != {key: params["source"][key] for key in ("repository", "revision")}:
        raise ValueError("Manifest differs from the source")
    if manifest["splits"] != report["splits"] or manifest["grouping"] != report["grouping"]:
        raise ValueError("Manifest differs from the prepared rows")
    report["great_expectations"] = great_expectations_checks(assigned)
    report["deepchecks"] = deepchecks_checks(assigned)
    report["success"] = True
    report["not_applicable"] = {
        "pynblint": "No notebooks in this data feature",
        "mlflow": "No model training in this data feature",
        "codecarbon": "Training measurements belong to the model feature",
    }
    write_json(report_path, report)
    return report
