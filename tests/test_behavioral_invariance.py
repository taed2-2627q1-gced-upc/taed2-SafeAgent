import copy

import pytest

from tests.behavioral_common import (
    load_behavioral_model,
    load_validation_rows_by_id,
    predict_behavioral_labels,
)

INVARIANCE_CASES = (
    (
        "allow_read_config",
        "004ce1ea5db5148d4606fc148831833a4123ae71f7d2b8eae8e09e724b0b5949",
        "ALLOW",
    ),
    (
        "allow_python_version",
        "01fc63fe52c29db8f1a41fb08860b0c9c78e8ecc581217ad8b834143397d0055",
        "ALLOW",
    ),
    (
        "ask_fetch_api_config",
        "007c84678056ba7d0b150f50ddea55629cf523fa2d308acc9780f3af984fd49a",
        "ASK",
    ),
    (
        "ask_test_production_site",
        "00843532a9127530a8320ce069b364e0d7e2ce56e5263cc99648fbd10ad3696c",
        "ASK",
    ),
    (
        "deny_publish_secret",
        "00cbb4d433625f01dd2c13e6724925c59298437250975e1f0d05a14c4736859d",
        "DENY",
    ),
    (
        "deny_disable_firewall",
        "01503195c56875fca7ca1d82282049a526da92a65863158f9521ff3df42b2cd6",
        "DENY",
    ),
)
UNRELATED_CONTEXT_MARKER = "invariance-unrelated-marker.txt"


@pytest.fixture(name="invariance_predictions", scope="module")
def fixture_invariance_predictions():
    tokenizer, model, metadata = load_behavioral_model()
    rows_by_id = load_validation_rows_by_id()
    missing_ids = {identifier for _, identifier, _ in INVARIANCE_CASES} - rows_by_id.keys()
    if missing_ids:
        raise AssertionError(
            f"Behavioral examples missing from validation data: {sorted(missing_ids)}"
        )

    original_rows = [rows_by_id[identifier] for _, identifier, _ in INVARIANCE_CASES]
    altered_rows = []
    for row in original_rows:
        altered = copy.deepcopy(row)
        untracked = altered["context"]["gitStatus"]["untracked"]
        if UNRELATED_CONTEXT_MARKER in untracked:
            raise ValueError(f"Context marker already exists in example: {row['id']}")
        untracked.append(UNRELATED_CONTEXT_MARKER)
        untracked.sort()
        altered_rows.append(altered)

    predicted_labels = predict_behavioral_labels(
        original_rows + altered_rows, tokenizer, model, metadata
    )
    return {
        case_name: (predicted_labels[index], predicted_labels[index + len(INVARIANCE_CASES)])
        for index, (case_name, _, _) in enumerate(INVARIANCE_CASES)
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case_name", "example_id", "expected_label"),
    INVARIANCE_CASES,
    ids=[case_name for case_name, _, _ in INVARIANCE_CASES],
)
def test_contextual_model_is_invariant_to_unrelated_untracked_file(
    invariance_predictions,
    case_name,
    example_id,
    expected_label,
):
    original_prediction, altered_prediction = invariance_predictions[case_name]
    assert original_prediction == expected_label, (
        f"{case_name} ({example_id}): expected original prediction {expected_label}, "
        f"got {original_prediction}"
    )
    assert altered_prediction == expected_label, (
        f"{case_name} ({example_id}): expected prediction {expected_label} after adding "
        f"{UNRELATED_CONTEXT_MARKER!r} to untracked files, got {altered_prediction}"
    )
