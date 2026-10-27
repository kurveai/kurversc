"""Experimental conditional target distributions for numeric regression.

Fit target bins using training labels only, learn their probabilities, then
derive the desired point statistic. Median and mean predictions share exactly
the same fitted model. No schema/task-specific thresholds or query calibration.
"""

import numpy as np


class OrdinalDistributionRegressor:
    def __init__(
        self,
        *,
        bins=32,
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_state=0,
        thread_count=4,
    ):
        if bins < 2 or min(iterations, depth, thread_count) < 1 or learning_rate <= 0:
            raise ValueError(
                "Positive model budgets and at least two bins are required"
            )
        self.bins, self.iterations, self.depth = bins, iterations, depth
        if l2_leaf_reg < 0:
            raise ValueError("L2 leaf regularization must be nonnegative")
        self.l2_leaf_reg = l2_leaf_reg
        self.learning_rate, self.random_state, self.thread_count = (
            learning_rate,
            random_state,
            thread_count,
        )

    def fit(self, x, y, *, cat_features=None):
        from catboost import CatBoostClassifier

        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) != len(x) or not len(y) or not np.isfinite(y).all():
            raise ValueError(
                "Finite, aligned one-dimensional training targets are required"
            )
        self.cutpoints_ = np.unique(
            np.quantile(y, np.linspace(0, 1, self.bins + 1)[1:-1])
        )
        codes = np.searchsorted(self.cutpoints_, y, side="left")
        self.classes_ = np.unique(codes)
        self.medians_ = np.array([np.median(y[codes == c]) for c in self.classes_])
        self.means_ = np.array([np.mean(y[codes == c]) for c in self.classes_])
        self.model_ = None
        if len(self.classes_) > 1:
            self.model_ = CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                loss_function="MultiClass",
                random_seed=self.random_state,
                thread_count=self.thread_count,
                verbose=False,
                allow_writing_files=False,
            )
            self.model_.fit(x, codes, cat_features=cat_features)
            np.testing.assert_array_equal(self.model_.classes_, self.classes_)
        self.fit_audit_ = dict(
            training_rows=len(y),
            requested_bins=self.bins,
            actual_bins=len(self.classes_),
            training_cutpoints=self.cutpoints_.tolist(),
            bin_counts=[int((codes == c).sum()) for c in self.classes_],
            bin_medians=self.medians_.tolist(),
            bin_means=self.means_.tolist(),
        )
        return self

    def predict_distribution(self, x):
        if not hasattr(self, "classes_"):
            raise ValueError("Fit the target distribution before predicting")
        if self.model_ is None:
            return np.ones((len(x), 1))
        return self.model_.predict_proba(x)

    def predict(self, x, *, statistic="median"):
        if statistic not in {"median", "mean"}:
            raise ValueError("Statistic must be median or mean")
        probability = self.predict_distribution(x)
        if statistic == "mean":
            return probability @ self.means_
        crossing = (probability.cumsum(axis=1) >= 0.5).argmax(axis=1)
        return self.medians_[crossing]
