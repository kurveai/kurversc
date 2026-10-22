"""Experimental bounded continuous-distribution readout; no graph changes."""

import numpy as np


class JointQuantileRegressor:
    """Learn a fixed, small quantile vector with one shared tree ensemble.

    Targets remain continuous. The median is used for MAE, and the additional
    heads supply distributional supervision without adding graph candidates.
    No validation labels or query statistics enter training or prediction.
    """

    def __init__(
        self,
        *,
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10.0,
        random_state=0,
        thread_count=4,
    ):
        levels = np.asarray(quantiles, dtype=float)
        if (
            levels.ndim != 1
            or len(levels) < 2
            or not np.isfinite(levels).all()
            or not ((levels > 0) & (levels < 1)).all()
            or not (np.diff(levels) > 0).all()
            or np.count_nonzero(levels == 0.5) != 1
        ):
            raise ValueError("Strictly increasing finite quantiles must include 0.5")
        if (
            min(iterations, depth, thread_count) < 1
            or learning_rate <= 0
            or l2_leaf_reg < 0
        ):
            raise ValueError("Positive model budgets and nonnegative L2 required")
        self.quantiles = tuple(levels.tolist())
        self.iterations, self.depth = iterations, depth
        self.learning_rate, self.l2_leaf_reg = learning_rate, l2_leaf_reg
        self.random_state, self.thread_count = random_state, thread_count

    def fit(self, x, y, *, cat_features=None):
        from catboost import CatBoostRegressor

        target = np.asarray(y, dtype=float)
        if (
            target.ndim != 1
            or len(target) != len(x)
            or not len(target)
            or not np.isfinite(target).all()
        ):
            raise ValueError("Finite aligned one-dimensional training targets required")
        self.model_ = None
        self.constant_ = float(target[0]) if np.unique(target).size == 1 else None
        if self.constant_ is None:
            self.model_ = CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                loss_function="MultiQuantile:alpha=" + ",".join(map(str, self.quantiles)),
                random_seed=self.random_state,
                thread_count=self.thread_count,
                verbose=False,
                allow_writing_files=False,
            ).fit(x, target, cat_features=cat_features)
        self.fit_audit_ = {
            "training_rows": len(target),
            "quantiles": list(self.quantiles),
            "objective": "shared-tree continuous multi-quantile loss",
            "point_statistic": "median",
        }
        return self

    def predict_quantiles(self, x, *, project=False):
        if not hasattr(self, "model_"):
            raise ValueError("Fit the quantile model before predicting")
        if self.model_ is None:
            return np.full((len(x), len(self.quantiles)), self.constant_, dtype=float)
        if not len(x):
            return np.empty((0, len(self.quantiles)), dtype=float)
        values = np.asarray(self.model_.predict(x), dtype=float)
        if project:
            from scipy.optimize import isotonic_regression

            values = np.array(
                [isotonic_regression(row, increasing=True).x for row in values]
            )
        return values

    def predict(self, x, *, project=False):
        return self.predict_quantiles(x, project=project)[:, self.quantiles.index(0.5)]
