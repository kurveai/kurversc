"""Experimental conditional-CDF regression with shared ordered thresholds."""

import numpy as np

from .ordinal_regression import OrdinalDistributionRegressor


class OrderedTargetRegressor(OrdinalDistributionRegressor):
    """Share one tree ensemble across Bernoulli events Y <= each target bin.

    Unlike nominal multiclass cross entropy, every ordered threshold carries
    supervision. Isotonic projection at inference restores a valid CDF without
    query labels or query-distribution calibration. Bins/representative values
    follow the same training-only recipe as the nominal comparison model.
    """

    def fit(self, x, y, *, cat_features=None):
        from catboost import CatBoostClassifier

        target = np.asarray(y, dtype=float)
        if (
            target.ndim != 1
            or len(target) != len(x)
            or not len(target)
            or not np.isfinite(target).all()
        ):
            raise ValueError("Finite aligned one-dimensional training targets required")
        self.cutpoints_ = np.unique(
            np.quantile(target, np.linspace(0, 1, self.bins + 1)[1:-1])
        )
        codes = np.searchsorted(self.cutpoints_, target, side="left")
        self.classes_ = np.unique(codes)
        self.medians_ = np.array([np.median(target[codes == c]) for c in self.classes_])
        self.means_ = np.array([np.mean(target[codes == c]) for c in self.classes_])
        self.threshold_classes_ = self.classes_[:-1]
        self.model_ = None
        if len(self.threshold_classes_):
            labels = (codes[:, None] <= self.threshold_classes_[None, :]).astype(int)
            if len(self.threshold_classes_) == 1:
                labels = labels[:, 0]
            self.model_ = CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                loss_function="Logloss"
                if len(self.threshold_classes_) == 1
                else "MultiLogloss",
                random_seed=self.random_state,
                thread_count=self.thread_count,
                verbose=False,
                allow_writing_files=False,
            ).fit(x, labels, cat_features=cat_features)
        self.fit_audit_ = dict(
            training_rows=len(target),
            requested_bins=self.bins,
            actual_bins=len(self.classes_),
            threshold_count=len(self.threshold_classes_),
            training_cutpoints=self.cutpoints_.tolist(),
            bin_counts=[int((codes == c).sum()) for c in self.classes_],
            bin_medians=self.medians_.tolist(),
            bin_means=self.means_.tolist(),
            objective="shared-tree binary threshold logloss",
            projection="unweighted isotonic CDF per example",
        )
        return self

    def predict_distribution(self, x):
        from scipy.optimize import isotonic_regression

        if not hasattr(self, "classes_"):
            raise ValueError("Fit the target distribution before predicting")
        if self.model_ is None:
            return np.ones((len(x), 1))
        probability = self.model_.predict_proba(x)
        if len(self.threshold_classes_) == 1:
            cdf = probability[:, 1, None]
        else:
            cdf = np.asarray(probability)
        cdf = np.clip(cdf, 0, 1)
        if cdf.shape[1] > 1:
            cdf = np.array([isotonic_regression(row, increasing=True).x for row in cdf])
        edges = np.column_stack([np.zeros(len(x)), cdf, np.ones(len(x))])
        mass = np.clip(np.diff(edges, axis=1), 0, 1)
        return mass / mass.sum(axis=1, keepdims=True)
