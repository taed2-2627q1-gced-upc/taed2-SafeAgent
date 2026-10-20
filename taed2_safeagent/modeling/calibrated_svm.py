"""Safety-calibrated Linear Support Vector Classifier."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC


# pylint: disable=invalid-name,too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
class SafetyCalibratedLinearSVC(BaseEstimator, ClassifierMixin):
    """Linear Support Vector Classifier with asymmetric safety threshold calibration.

    In shell-agent security, allowing a destructive command (False ALLOW) poses
    a critical vulnerability, whereas asking for human confirmation (False ASK)
    is only an inconvenience. This classifier applies a configurable positive
    decision offset to the DENY class score before taking the argmax,
    significantly boosting safety recall while preserving sub-millisecond latency.
    """

    def __init__(
        self,
        C=1.0,
        class_weight="balanced",
        deny_boost=0.4,
        random_state=42,
        dual="auto",
        tol=0.0001,
        max_iter=10000,
    ):
        self.C = C
        self.class_weight = class_weight
        self.deny_boost = deny_boost
        self.random_state = random_state
        self.dual = dual
        self.tol = tol
        self.max_iter = max_iter
        self.model_ = None
        self.classes_ = None
        self.deny_idx_ = None
        self.n_iter_ = None

    def fit(self, X, y):
        """Fit the underlying LinearSVC model and identify the DENY class index."""
        self.model_ = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            dual=self.dual,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_iter_ = self.model_.n_iter_
        classes_list = list(self.classes_)
        if "DENY" in classes_list:
            self.deny_idx_ = classes_list.index("DENY")
        else:
            self.deny_idx_ = None
        return self

    def decision_function(self, X):
        """Compute raw decision function scores from the underlying LinearSVC."""
        return self.model_.decision_function(X)

    def predict(self, X):
        """Predict class labels with asymmetric safety margin applied to DENY."""
        scores = np.asarray(self.model_.decision_function(X)).copy()
        if self.deny_boost and self.deny_idx_ is not None and scores.ndim > 1:
            scores[:, self.deny_idx_] += self.deny_boost
            best_idx = np.argmax(scores, axis=1)
            return self.classes_[best_idx]
        return self.model_.predict(X)
