"""Unit tests for SafetyCalibratedLinearSVC."""

# pylint: disable=invalid-name

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from taed2_safeagent.modeling.calibrated_svm import SafetyCalibratedLinearSVC


def test_calibrated_svm_initialization():
    clf = SafetyCalibratedLinearSVC(C=2.0, deny_boost=0.5, random_state=123)
    assert clf.C == 2.0
    assert clf.deny_boost == 0.5
    assert clf.random_state == 123
    assert clf.classes_ is None


def test_calibrated_svm_fit_and_classes():
    X = ["cat file.txt", "rm -rf /", "git push", "chmod 777 script.sh"]
    y = ["ALLOW", "DENY", "ASK", "DENY"]
    clf = SafetyCalibratedLinearSVC(C=1.0, deny_boost=0.4, random_state=42)
    vec = CountVectorizer()
    X_vec = vec.fit_transform(X)
    clf.fit(X_vec, y)

    assert clf.classes_ is not None
    assert set(clf.classes_) == {"ALLOW", "ASK", "DENY"}
    assert clf.deny_idx_ == list(clf.classes_).index("DENY")
    assert clf.n_iter_ is not None


def test_calibrated_svm_boost_shifts_decision():
    # Construct a dataset where one sample is near the boundary
    X = [
        "ls -la",
        "rm file.txt",
        "sudo rm -rf /",
        "apt update",
    ]
    y = ["ALLOW", "ASK", "DENY", "ASK"]
    vec = CountVectorizer()
    X_vec = vec.fit_transform(X)

    uncalibrated = SafetyCalibratedLinearSVC(C=1.0, deny_boost=0.0, random_state=42)
    uncalibrated.fit(X_vec, y)
    raw_scores = uncalibrated.decision_function(X_vec)

    calibrated = SafetyCalibratedLinearSVC(C=1.0, deny_boost=1.5, random_state=42)
    calibrated.fit(X_vec, y)
    preds = calibrated.predict(X_vec)

    # With high deny_boost, DENY predictions must be at least as frequent as uncalibrated
    deny_uncal = sum(p == "DENY" for p in uncalibrated.predict(X_vec))
    deny_cal = sum(p == "DENY" for p in preds)
    assert deny_cal >= deny_uncal
    np.testing.assert_array_equal(calibrated.decision_function(X_vec), raw_scores)


def test_calibrated_svm_pipeline_and_joblib_roundtrip(tmp_path):
    pipe = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("svm", SafetyCalibratedLinearSVC(C=1.0, deny_boost=0.4, random_state=42)),
        ]
    )
    X = ["cat file.txt", "rm -rf /", "git commit"]
    y = ["ALLOW", "DENY", "ASK"]
    pipe.fit(X, y)

    preds_original = pipe.predict(X)
    assert list(pipe.classes_) == list(pipe.named_steps["svm"].classes_)

    model_file = tmp_path / "calibrated_pipeline.joblib"
    joblib.dump(pipe, model_file)

    loaded_pipe = joblib.load(model_file)
    preds_loaded = loaded_pipe.predict(X)

    np.testing.assert_array_equal(preds_original, preds_loaded)
    assert list(loaded_pipe.classes_) == list(pipe.classes_)
