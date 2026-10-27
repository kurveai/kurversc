"""Learner-stage, entity-crossfitted semantic risk annotation; not a causal relation.

All training labels are mature at the outer fitting boundary. Each training
entity's annotations exclude every label for that entity. The full classifier
only annotates held-out covariates. No validation targets or root IDs are inputs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.benchmark_reliability_blending import group_masks


class CrossfitSemanticPropensity:
    def fit(self, features, labels, *, entities, timestamps, origin, horizon):
        x = pd.DataFrame(features).astype("float64")
        y, entities = np.asarray(labels), np.asarray(entities)
        times = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if (not len(x) == len(y) == len(entities) == len(times) or not 1 <= len(x.columns) <= 32
                or x.columns.duplicated().any() or any(c.startswith("__") for c in x)
                or not np.isfinite(x).all().all() or pd.isna(entities).any() or times.hasnans
                or set(np.unique(y)) != {0, 1}):
            raise ValueError("Aligned bounded finite semantic payload and binary mature labels required")
        if pd.Timedelta(horizon) <= pd.Timedelta(0) or (times + pd.Timedelta(horizon) > pd.Timestamp(origin)).any():
            raise ValueError("All learner-stage labels must have matured at the fit boundary")
        self.columns = list(x)
        self.oof, covered, audits = np.empty(len(x)), np.zeros(len(x), dtype="int8"), []
        for number, (fit, query) in enumerate(group_masks(entities)):
            if set(entities[fit]) & set(entities[query]) or set(np.unique(y[fit])) != {0, 1}:
                raise ValueError("Entity-disjoint complete binary fitting groups required")
            model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-8, random_state=0)
            model.fit(x.loc[fit], y[fit])
            self.oof[query], covered[query] = model.decision_function(x.loc[query]), covered[query] + 1
            audits.append(dict(group=number, training_rows=int(fit.sum()), annotation_rows=int(query.sum()),
                own_entity_labels_excluded=True))
        np.testing.assert_array_equal(covered, np.ones(len(x)))
        self.model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-8, random_state=0).fit(x, y)
        self.audit = dict(dimensions=len(x.columns), columns=self.columns, C=1., solver="lbfgs", class_weight=None,
            groups=audits, input_is_frozen_pretrained_projection=True, structural_keys_are_predictors=False,
            latest_training_label_complete_at=str((times + pd.Timedelta(horizon)).max()), origin=str(origin),
            scope="Supervised learner-stage crossfit; NOT a pre-query historical label feature")
        return self

    def transform(self, features):
        x = pd.DataFrame(features).astype("float64")
        if list(x) != self.columns or not np.isfinite(x).all().all():
            raise ValueError("Same ordered finite semantic covariates required")
        return self.model.decision_function(x)


def bounded_joint_annotation(frame, probability_logit):
    """Same payload width: replace the last variance-ranked basis coordinate."""
    frame = pd.DataFrame(frame).reset_index(drop=True)
    values = np.asarray(probability_logit)
    if len(frame) != len(values) or not 2 <= len(frame.columns) <= 32 or not np.isfinite(values).all():
        raise ValueError("Aligned bounded joint semantic annotation required")
    result = frame.iloc[:, :-1].copy()
    result["semantic_propensity_crossfit_logit"] = values.astype("float32")
    return result
