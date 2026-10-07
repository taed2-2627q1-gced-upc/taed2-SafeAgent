from copy import deepcopy
import json
from pathlib import Path
import warnings

import joblib
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from typer.testing import CliRunner
import yaml

from taed2_safeagent.data.common import LABELS, read_jsonl, write_json, write_jsonl
from taed2_safeagent.features import character_ngrams, make_vectorizer
from taed2_safeagent.modeling import evaluate as evaluation_cli
from taed2_safeagent.modeling import train as training_cli
from taed2_safeagent.modeling.common import load_config, load_examples, load_model
from taed2_safeagent.modeling.evaluate import compute_metrics, evaluate
from taed2_safeagent.modeling.train import fit_pipeline, train


def test_character_features_preserve_case_spaces_and_symbols():
    assert character_ngrams("A  B", (2, 2)) == ["A ", "  ", " B"]
    assert character_ngrams("rm;X", (2, 2)) == ["rm", "m;", ";X"]
    assert character_ngrams("Ab", (2, 5)) == ["Ab"]


def test_vocabulary_is_fitted_on_training_text_only(baseline_params):
    settings = deepcopy(baseline_params["baseline"])
    settings["tfidf"]["min_df"] = 1
    texts = ["allow aaa", "ask bbb", "deny ccc"]
    pipeline = fit_pipeline(texts, list(LABELS), settings)
    vocabulary = dict(pipeline.named_steps["tfidf"].vocabulary_)
    pipeline.predict(["ZZZ validation only"])
    assert pipeline.named_steps["tfidf"].vocabulary_ == vocabulary
    assert "ZZ" not in vocabulary


def test_roundtrip_and_evaluation_keep_ids_aligned(baseline_params):
    metadata = train(baseline_params)
    pipeline, stored = load_model(baseline_params["baseline"]["model_dir"])
    assert stored == metadata
    examples = load_examples(baseline_params, "validation")
    expected = pipeline.predict([row["command"] for row in examples]).tolist()
    metrics = evaluate(baseline_params)
    predictions_path = baseline_params["baseline"]["report_dir"] / "predictions.jsonl"
    predictions = [row for _, row in read_jsonl(predictions_path)]
    assert [row["predicted_label"] for row in predictions] == expected
    assert [row["id"] for row in predictions] == [row["id"] for row in examples]
    assert all(set(row) == {"id", "true_label", "predicted_label"} for row in predictions)
    assert metrics["split"] == "validation"
    assert metrics["rows"] == len(examples)
    assert metrics["confusion_matrix"]["labels"] == list(LABELS)


def test_targets_and_metadata_do_not_enter_vocabulary(baseline_params):
    model_dir = baseline_params["baseline"]["model_dir"]
    train(baseline_params)
    pipeline, _ = load_model(model_dir)
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    assert "allo" not in vocabulary
    assert "DENY" not in vocabulary
    assert "local" not in vocabulary
    assert "posix" not in vocabulary


def test_metrics_count_safety_errors_and_keep_matrix_order():
    metrics = compute_metrics(["DENY", "DENY", "ASK", "ALLOW"],
                              ["ALLOW", "DENY", "ALLOW", "ALLOW"])
    assert metrics["deny_to_allow_count"] == 1
    assert metrics["deny_to_allow_rate"] == 0.5
    assert metrics["classes"]["DENY"]["recall"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(7 / 18)
    assert metrics["confusion_matrix"]["counts"] == [[1, 0, 0], [1, 0, 0], [1, 0, 1]]


@pytest.mark.parametrize("expected,predicted", [([], []), (["ASK"], []),
                                             (["unknown"], ["ASK"]), (["ASK"], ["unknown"])])
def test_invalid_metrics_fail(expected, predicted):
    with pytest.raises(ValueError):
        compute_metrics(expected, predicted)


def test_no_deny_examples_has_no_false_allow_rate():
    assert compute_metrics(["ALLOW", "ASK"], ["ALLOW", "ASK"])["deny_to_allow_rate"] is None


def test_empty_vocabulary_does_not_write_a_model(baseline_params):
    baseline_params["baseline"]["tfidf"]["min_df"] = 10000
    with pytest.raises(ValueError):
        train(baseline_params)
    assert not baseline_params["baseline"]["model_dir"].exists()


def test_nonconvergence_fails_without_writing_a_model(baseline_params, monkeypatch):
    def not_converged(_self, _texts, _labels):
        warnings.warn("Not converged", ConvergenceWarning)

    monkeypatch.setattr(LinearSVC, "fit", not_converged)
    with pytest.raises(ValueError, match="converge"):
        train(baseline_params)
    assert not baseline_params["baseline"]["model_dir"].exists()


def test_missing_model_and_reserved_test_fail(baseline_params):
    with pytest.raises(FileNotFoundError):
        evaluate(baseline_params)
    with pytest.raises(ValueError, match="reserved"):
        load_examples(baseline_params, "test")


def test_invalid_targets_fail(baseline_params):
    path = baseline_params["paths"]["processed"] / "train.jsonl"
    rows = [row for _, row in read_jsonl(path)]
    rows[0]["label"] = "UNKNOWN"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="label"):
        train(baseline_params)


def test_changed_training_data_invalidates_evaluation(baseline_params):
    train(baseline_params)
    path = baseline_params["paths"]["processed"] / "train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="training dataset"):
        evaluate(baseline_params)


def test_overlapping_groups_and_stale_reports_are_rejected(baseline_params):
    train(baseline_params)
    training = load_examples(baseline_params, "train")
    path = baseline_params["paths"]["processed"] / "validation.jsonl"
    rows = [row for _, row in read_jsonl(path)]
    rows[0]["group_id"] = training[0]["group_id"]
    write_jsonl(path, rows)
    report = baseline_params["baseline"]["report_dir"] / "metrics.json"
    write_json(report, {"accuracy": 1})
    with pytest.raises(ValueError, match="overlap"):
        evaluate(baseline_params)
    assert not report.exists()


def test_cli_propagates_errors(tmp_path):
    runner = CliRunner()
    for app in (training_cli.app, evaluation_cli.app):
        assert runner.invoke(app, ["--help"]).exit_code == 0
        result = runner.invoke(app, ["--params", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1
        assert "failed" in result.output


@pytest.mark.parametrize(
    "field,value",
    [("model_dir", "data/raw/bad"), ("report_dir", "../outside"),
     ("input_mode", "command-context")],
)
def test_invalid_config_paths_and_input_mode_fail(tmp_path, field, value):
    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "params.yaml").read_text(encoding="utf-8"))
    settings["baseline"][field] = value
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_character_range_fails(baseline_params):
    settings = dict(baseline_params["baseline"]["tfidf"], ngram_range=[5, 2])
    with pytest.raises(ValueError, match="range"):
        make_vectorizer(settings)


def test_model_metadata_is_checked(baseline_params):
    train(baseline_params)
    path = baseline_params["baseline"]["model_dir"] / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["input_mode"] = "command-context"
    write_json(path, metadata)
    with pytest.raises(ValueError, match="input mode"):
        load_model(path.parent)


def test_pipeline_roundtrip_keeps_predictions(baseline_params, tmp_path):
    settings = deepcopy(baseline_params["baseline"])
    settings["tfidf"]["min_df"] = 1
    texts = ["git status", "pip install", "erase disk"]
    pipeline = fit_pipeline(texts, list(LABELS), settings)
    expected = pipeline.predict(texts).tolist()
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    assert joblib.load(path).predict(texts).tolist() == expected
