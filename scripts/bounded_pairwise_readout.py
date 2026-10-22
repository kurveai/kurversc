"""Experimental binary ranking head with bounded pairs and training calibration."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def bounded_positive_pairs(labels, *, maximum=100000, seed=0):
    y = np.asarray(labels)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1} or not 1 <= maximum <= 100000:
        raise ValueError("Binary labels with both classes and a bounded pair budget required")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    total = len(positive) * len(negative)
    if total <= maximum:
        return np.column_stack((np.repeat(positive, len(negative)), np.tile(negative, len(positive))))
    rng = np.random.default_rng(seed)
    # Unique flat indices give a deterministic, label-stratified pair sample.
    chosen = rng.choice(total, size=maximum, replace=False)
    return np.column_stack((positive[chosen // len(negative)], negative[chosen % len(negative)]))


class BoundedPairwiseReadout:
    def __init__(self, *, seed=0, model_params=None):
        self.seed, self.model_params = seed, dict(model_params or {})

    def fit(self, features, labels, *, categorical):
        from catboost import CatBoostRanker

        y = np.asarray(labels, dtype=int)
        if len(features) != len(y):
            raise ValueError("Aligned feature/target rows required")
        params = {k: v for k, v in self.model_params.items() if k != "auto_class_weights"}
        params.update(loss_function="PairLogit", random_seed=self.seed,
                      verbose=False, allow_writing_files=False)
        pairs = bounded_positive_pairs(y, maximum=100000, seed=self.seed + 61009)
        self.model = CatBoostRanker(**params).fit(features, y, cat_features=categorical,
            group_id=np.zeros(len(y), dtype="int32"), pairs=pairs)
        training_scores = np.asarray(self.model.predict(features)).reshape(-1, 1)
        prevalence = float(y.mean())
        ratio = np.sqrt((1 - prevalence) / prevalence)
        self.calibrator = LogisticRegression(C=1., solver="liblinear", max_iter=1000,
            random_state=self.seed, class_weight={0: 1., 1: ratio}).fit(training_scores, y)
        self.audit = dict(training_rows=len(y), positive_prior=prevalence, pairs=len(pairs),
                          maximum_pairs=100000, calibration="training-only sqrt-balanced logistic",
                          calibration_weight_ratio=ratio, objective="PairLogit")
        return self

    def predict_proba(self, features):
        values = np.asarray(self.model.predict(features)).reshape(-1, 1)
        return self.calibrator.predict_proba(values)


def fit_pairwise_policy(features, labels, timestamps, *, origin, label_horizon,
                        refit_frames, policy, task, categorical, random_state=0,
                        row_limit=100000, model_params=None):
    """Same temporal support/sampling as the existing frame-ensemble fitter."""
    from dataclasses import asdict
    import pandas as pd
    from kurversc.modeling import FrameEnsemble, sample_training_rows
    from kurversc.temporal_policy import policy_support

    if task != "classification" or policy.mode != "frame_ensemble" or policy.blend_weight != 1:
        raise ValueError("Unblended binary frame-ensemble support required")
    if not (len(features) == len(labels) == len(timestamps)):
        raise ValueError("Aligned temporal cohorts required")
    x, y = features.reset_index(drop=True), labels.reset_index(drop=True)
    times = pd.to_datetime(timestamps).reset_index(drop=True)
    support = policy_support(pd.DataFrame({"cutoff": times}), timestamp="cutoff",
        origin=origin, label_horizon=label_horizon, refit_frames=refit_frames, policy=policy)
    params = dict(iterations=300, depth=6, learning_rate=.05, l2_leaf_reg=10., thread_count=2)
    params.update(model_params or {})
    models, fitted_rows, audits = [], 0, []
    for number, cutoff in enumerate(support["cutoffs"], start=1):
        mask = times.eq(cutoff)
        tx, ty = sample_training_rows(x.loc[mask], y.loc[mask], limit=row_limit,
            task="classification", random_state=random_state + number)
        model = BoundedPairwiseReadout(seed=random_state, model_params=params).fit(tx, ty,
                                                                              categorical=categorical)
        models.append(model)
        fitted_rows += len(tx)
        audits.append(model.audit)
    return FrameEnsemble(models, task="classification"), dict(policy=asdict(policy),
        origin=str(pd.Timestamp(origin)), initial_cutoffs=list(map(str, support["initial"])),
        observed_refit_cutoffs=list(map(str, support["refit"])), training_rows=fitted_rows,
        skipped_single_class_cutoffs=[], member_audits=audits)
