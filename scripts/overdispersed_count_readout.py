"""One fixed-budget count head with a training-estimated overdispersion.

A negative-binomial likelihood is a weighted soft-label cross entropy when
its dispersion is fixed: target=c/(c+r), weight=c+r. This avoids custom Python
objectives and a graph search expansion. r is estimated from TRAINING counts
only. Prediction returns the probability above the exactly identified task
threshold, never a count or future response as an input feature.
"""

import numpy as np
from scipy.special import betainc


def training_dispersion(counts):
    values = np.asarray(counts, float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or (values < 0).any() or not np.equal(values, np.floor(values)).all()):
        raise ValueError("Finite nonnegative integral training counts required")
    mean, variance = float(values.mean()), float(values.var())
    if mean <= 0 or np.unique(values).size < 3:
        raise ValueError("A nondegenerate positive-mean count sample is required")
    return float(np.clip(mean * mean / max(variance - mean, 1e-12), .1, 100.))


class OverdispersedCountReadout:
    def __init__(self, *, threshold, polarity=1, seed=0, model_params=None, mode="cumulative"):
        if threshold < 0 or int(threshold) != threshold or polarity not in (0, 1) or mode != "cumulative":
            raise ValueError("Exactly identified threshold/polarity and one fixed count profile required")
        self.threshold, self.polarity, self.seed = int(threshold), polarity, seed
        self.params = dict(model_params or {})

    def fit(self, features, counts, *, categorical):
        from catboost import CatBoostClassifier

        values = np.asarray(counts, float)
        self.dispersion = training_dispersion(values)
        if len(features) != len(values):
            raise ValueError("Exact original feature/count alignment required")
        if set(self.params) - {"iterations", "depth", "learning_rate", "l2_leaf_reg", "thread_count"}:
            raise ValueError("Unsupported research learner parameters")
        options = dict(iterations=300, depth=6, learning_rate=.05, l2_leaf_reg=10., thread_count=4,
            random_seed=self.seed, verbose=False, allow_writing_files=False)
        options.update(self.params)
        weights = values + self.dispersion
        targets = values / weights
        self.model = CatBoostClassifier(loss_function="CrossEntropy", **options).fit(
            features, targets, cat_features=categorical, sample_weight=weights / weights.mean())
        self.audit = dict(strategy="fixed-dispersion negative-binomial count likelihood",
            dispersion=self.dispersion, dispersion_selection="Clipped training-only method of moments [.1,100]",
            rows=len(features), columns=len(features.columns), models=1, output_dimensions=1,
            trees=options["iterations"], task_threshold=self.threshold, polarity=self.polarity,
            auxiliary_counts_are_predictors=False, original_supervised_rows_unchanged=True,
            compute_note="Same scalar tree cap; different supervised likelihood, not a new family")
        return self

    def predict_proba(self, features):
        probability = np.asarray(self.model.predict_proba(features), float)[:, 1]
        positive = betainc(self.threshold + 1, self.dispersion, probability)
        if self.polarity == 0:
            positive = 1 - positive
        if not np.isfinite(positive).all() or (positive < 0).any() or (positive > 1).any():
            raise ValueError("Finite count-tail probabilities required")
        return np.column_stack([1 - positive, positive])
