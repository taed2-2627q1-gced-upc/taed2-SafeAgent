import pytest

from tests.behavioral_common import (
    load_behavioral_model,
    load_validation_rows_by_id,
    predict_behavioral_labels,
)

MINIMUM_FUNCTIONALITY_CASES = (
    (
        "allow_git_diff",
        "0a8a5ae24279eb6a13c1fae32cb637d8fcb89ea690a7d457b35c61a364335b62",
        "ALLOW",
    ),
    (
        "allow_show_commit",
        "4e6b55a67374e97fffa10be6a5bab4606e003b9034d38e000873ab0414d157aa",
        "ALLOW",
    ),
    (
        "allow_recent_git_history",
        "6efd97750d3e43526d5ba1a0d48c47fccc058be6ccddd52f59267afe0f89f25d",
        "ALLOW",
    ),
    (
        "ask_apply_staging_ingress",
        "91431a62210c4ff3cdcada01be9d2491d5f0685ac5a50d3e9fbefb8b3917d1e3",
        "ASK",
    ),
    (
        "ask_restart_production_pods",
        "5683ec95886f22db42abbc33bebc69156fa04b50ca694a4a2a1616364ba6a61f",
        "ASK",
    ),
    (
        "ask_force_push_without_request",
        "03799874975daf124a902ccf87a839d685d45b48e915b18654c61e15fc031b34",
        "ASK",
    ),
    (
        "deny_remove_home_for_build_cleanup",
        "1a105ff55921a0f16c92568cc604d912cfbe7c6cbfe00286d74cda145cc12bf4",
        "DENY",
    ),
    (
        "deny_delete_secret_for_test_cleanup",
        "05ed83e012e29600d1f68bea32130fc7f56c27b15cb96d7ba730f5e3baee1b06",
        "DENY",
    ),
    (
        "deny_send_env_for_server_start",
        "01d2399160f471cc3238bf114a57d2f1646db0ea40b5dea9d89fba2139b91269",
        "DENY",
    ),
)


@pytest.fixture(name="minimum_functionality_predictions", scope="module")
def fixture_minimum_functionality_predictions():
    tokenizer, model, metadata = load_behavioral_model()
    rows_by_id = load_validation_rows_by_id()
    selected_ids = {identifier for _, identifier, _ in MINIMUM_FUNCTIONALITY_CASES}
    missing_ids = selected_ids - rows_by_id.keys()
    if missing_ids:
        raise AssertionError(
            f"Behavioral examples missing from validation data: {sorted(missing_ids)}"
        )

    selected_rows = [rows_by_id[identifier] for identifier in selected_ids]
    for row in selected_rows:
        context = row["context"]
        has_context = any(
            (
                context["gitRemote"],
                context["gitStatus"]["untracked"],
                context["gitStatus"]["modified"],
                context["gitStatus"]["staged"],
                context["agentTouchedFiles"],
                context["lastUserPrompt"],
                context["cwd"],
            )
        )
        if not has_context:
            raise AssertionError(f"Behavioral example has no informative context: {row['id']}")

    predicted_labels = predict_behavioral_labels(
        selected_rows, tokenizer, model, metadata
    )
    return {
        row["id"]: prediction
        for row, prediction in zip(selected_rows, predicted_labels)
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case_name", "example_id", "expected_label"),
    MINIMUM_FUNCTIONALITY_CASES,
    ids=[case_name for case_name, _, _ in MINIMUM_FUNCTIONALITY_CASES],
)
def test_contextual_model_minimum_functionality(
    minimum_functionality_predictions,
    case_name,
    example_id,
    expected_label,
):
    assert minimum_functionality_predictions[example_id] == expected_label, (
        f"{case_name}: expected {expected_label}, "
        f"got {minimum_functionality_predictions[example_id]}"
    )
