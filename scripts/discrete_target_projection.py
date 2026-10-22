"""Bounded training-only target-domain readout; isolated optional challenger.

No upper clipping: unseen positive outcomes may exceed any observed count.
Count projection is applicable only to variable, nonnegative integer training
targets. Continuous nonnegative targets may use lower-bound clipping only.
Validation outcomes never determine applicability, bounds, or lattice spacing.
"""

import numpy as np
import pandas as pd


class TargetDomainProjection:
    def fit(self, labels, timestamps, *, origin, horizon):
        y = np.asarray(labels, dtype=float)
        clocks = pd.to_datetime(pd.Series(timestamps))
        if len(y) != len(clocks) or not len(y) or not np.isfinite(y).all() or clocks.isna().any():
            raise ValueError("Complete aligned mature training targets required")
        duration, boundary = pd.Timedelta(horizon), pd.Timestamp(origin)
        if duration <= pd.Timedelta(0) or (clocks >= boundary).any() or (clocks+duration > boundary).any():
            raise ValueError("Only labels matured by fit boundary are eligible")
        unique = np.unique(y)
        self.nonnegative = bool(np.all(y >= 0))
        self.integer_counts = bool(self.nonnegative and len(unique) >= 3
            and np.all(np.abs(y-np.rint(y)) <= 1e-10))
        self.audit = dict(training_rows=len(y), nonnegative=self.nonnegative,
            integer_counts=self.integer_counts, unique_observed_targets=len(unique),
            latest_training_label_complete_at=str((clocks+duration).max()),
            lower_bound=0 if self.nonnegative else None, upper_bound=None,
            support_rule="Training-only observed domain; optional validation gate still required")
        return self

    def predict_values(self, prediction, *, recipe):
        raw = np.asarray(prediction, dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Finite incumbent predictions required")
        if recipe == "nonnegative":
            if not self.nonnegative:
                raise ValueError("Nonnegative projection is inapplicable")
            return np.maximum(raw, 0)
        if recipe == "integer_counts":
            if not self.integer_counts:
                raise ValueError("Integer-count projection is inapplicable")
            return np.rint(np.maximum(raw, 0))
        raise ValueError("Undeclared projection recipe")
