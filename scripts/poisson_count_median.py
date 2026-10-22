"""Conditional count-median ablation without marginal-dispersion collapse."""

import numpy as np
from scipy.stats import poisson


class PoissonCountMedian:
    def __init__(self, *, seed=0, model_params=None):
        self.seed, self.params = seed, dict(model_params or {})

    def fit(self, features, labels, *, categorical):
        from catboost import CatBoostRegressor

        values = np.asarray(labels, float)
        if (values.shape != (len(features),) or not np.isfinite(values).all()
                or (values < 0).any() or not np.equal(values, np.floor(values)).all()
                or np.unique(values).size < 3):
            raise ValueError("Variable nonnegative integral original count targets required")
        if set(self.params)-{"iterations", "depth", "learning_rate", "l2_leaf_reg", "thread_count"}:
            raise ValueError("Unsupported research learner parameters")
        options = dict(iterations=300, depth=6, learning_rate=.05, l2_leaf_reg=10., thread_count=4,
            random_seed=self.seed, verbose=False, allow_writing_files=False)
        options.update(self.params)
        self.model = CatBoostRegressor(loss_function="Poisson", **options).fit(features, values, cat_features=categorical)
        self.audit = dict(strategy="conditional Poisson median", rows=len(values), models=1,
            trees=options["iterations"], output_dimensions=1, upper_target_clipping=False,
            marginal_variance_used_as_conditional_dispersion=False, no_added_predictors=True)
        return self

    def predict(self, features):
        log_mean = np.asarray(self.model.predict(features, prediction_type="RawFormulaVal"), float)
        if not np.isfinite(log_mean).all():
            raise ValueError("Finite conditional log-rates required")
        mean = np.exp(np.clip(log_mean, -30., 30.))
        prediction = poisson.ppf(.5, mean)
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError("Finite nonnegative conditional count medians required")
        return prediction
