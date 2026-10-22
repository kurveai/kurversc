"""Opt-in conditional count medians; no change to graph or fit defaults.

The caller must provide fully observed training targets and validate this
readout against the complete incumbent estimator. Integer-valued training
labels alone do not establish useful Poisson conditional assumptions. This
model is not automatically selected or enabled by ``kurversc.fit``.
"""

import numpy as np


class CountMedianRegressor:
    """A fixed-budget Poisson rate model decoded to a conditional median.

    Intended as a validation-gated alternative for count-target MAE, not as
    another GraphReduce feature family. Supports ordinary CatBoost categorical
    feature indices/names. There is no clipping to the largest training count.
    """

    def __init__(self, *, iterations=300, depth=6, learning_rate=.05,
                 l2_leaf_reg=10., random_state=0, thread_count=4):
        if (min(iterations, depth, thread_count) < 1 or not np.isfinite(learning_rate)
                or learning_rate <= 0 or not np.isfinite(l2_leaf_reg) or l2_leaf_reg < 0):
            raise ValueError("Positive model budgets and nonnegative finite L2 are required")
        self.iterations, self.depth, self.learning_rate = iterations, depth, learning_rate
        self.l2_leaf_reg, self.random_state, self.thread_count = l2_leaf_reg, random_state, thread_count

    def fit(self, x, y, *, cat_features=None):
        from catboost import CatBoostRegressor

        values = np.asarray(y, dtype=float)
        if (values.shape != (len(x),) or not np.isfinite(values).all() or (values < 0).any()
                or not np.equal(values, np.floor(values)).all() or np.unique(values).size < 3):
            raise ValueError("Aligned, finite, variable nonnegative integer training targets required")
        self.model_ = CatBoostRegressor(iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg, thread_count=self.thread_count,
            random_seed=self.random_state, loss_function="Poisson", verbose=False, allow_writing_files=False).fit(
                x, values, cat_features=cat_features)
        self.fit_audit_ = dict(strategy="conditional Poisson median", training_rows=len(values),
            trees=self.iterations, models=1, upper_target_clipping=False, added_predictors=0,
            marginal_variance_used_as_conditional_dispersion=False,
            applicability="Training target domain only; independent temporal validation still required",
            enabled_by_default=False)
        return self

    def predict(self, x):
        from scipy.stats import poisson

        if not hasattr(self, "model_"):
            raise ValueError("Fit the count-median regressor before predicting")
        if not len(x):
            return np.empty(0, dtype=float)
        log_mean = np.asarray(self.model_.predict(x, prediction_type="RawFormulaVal"), dtype=float)
        if log_mean.shape != (len(x),) or not np.isfinite(log_mean).all():
            raise ValueError("Finite row-aligned conditional log-rates required")
        prediction = poisson.ppf(.5, np.exp(np.clip(log_mean, -30., 30.)))
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError("Finite nonnegative conditional count medians required")
        return prediction
