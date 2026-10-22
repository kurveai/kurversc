"""Experimental, training-selected scale-aware absolute-error regression.

A fixed tree ensemble cannot extrapolate its leaf values linearly with size.
Learn a conditional rate and multiply by a positive, observed scale instead.
Exposure-weighted rate MAE is proportional to original-unit MAE. This does not
assume the target is a binomial count or that its rate is bounded by one.
"""

import numpy as np
import pandas as pd


class ExposureScaledRegressor:
    def __init__(
        self,
        *,
        weighted=True,
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_state=0,
        thread_count=4,
    ):
        self.weighted, self.iterations, self.depth = weighted, iterations, depth
        self.learning_rate, self.random_state, self.thread_count = (
            learning_rate,
            random_state,
            thread_count,
        )

    def _scale(self, exposures):
        if self.column_ is None:
            return np.ones(len(exposures))
        if self.column_ not in exposures:
            raise ValueError("The fitted scale input is missing")
        values = exposures[self.column_].to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        if (values[finite] < 0).any():
            raise ValueError("Scale inputs must stay nonnegative")
        values = np.where(finite, values, self.fill_)
        return 1.0 + values

    def fit(self, x, y, exposures, *, cat_features=None):
        from catboost import CatBoostRegressor

        target = np.asarray(y, dtype=float)
        if (
            target.ndim != 1
            or len(target) != len(x)
            or len(exposures) != len(x)
            or not len(x)
            or not np.isfinite(target).all()
            or not exposures.columns.is_unique
        ):
            raise ValueError("Finite aligned targets and unique scale inputs required")
        if isinstance(x, pd.DataFrame) and not x.index.equals(exposures.index):
            raise ValueError("Features and scale inputs must have identical indexes")
        self.column_, self.fill_ = None, 0.0
        candidates = []
        for column in sorted(exposures.columns):
            if not pd.api.types.is_numeric_dtype(exposures[column]):
                continue
            values = exposures[column].to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(values)
            if finite.mean() < 0.95 or (values[finite] < 0).any():
                continue
            observed = values[finite]
            low, high = np.quantile(observed, [0.1, 0.9])
            if len(np.unique(observed)) < 16 or (high + 1) / (low + 1) < 4:
                continue
            correlation = float(
                pd.Series(observed).corr(pd.Series(target[finite]), method="spearman")
            )
            if np.isfinite(correlation) and correlation > 0:
                candidates.append((correlation, column, float(np.median(observed))))
        if candidates:
            candidates.sort(key=lambda row: (-row[0], row[1]))
            _, self.column_, self.fill_ = candidates[0]
        scale = self._scale(exposures)
        rate = target / scale
        weights = scale / scale.mean() if self.weighted else None
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="MAE",
            random_seed=self.random_state,
            thread_count=self.thread_count,
            verbose=False,
            allow_writing_files=False,
        ).fit(x, rate, sample_weight=weights, cat_features=cat_features)
        self.fit_audit_ = dict(
            training_rows=len(target),
            selected_column=self.column_,
            missing_fill=self.fill_,
            weighted=self.weighted,
            candidate_ranks=[dict(correlation=c, column=n) for c, n, _ in candidates],
            scale_quantiles=np.quantile(scale, [0, 0.5, 0.9, 0.99, 1]).tolist(),
            objective="original-unit MAE" if self.weighted else "rate MAE diagnostic",
            fallback=self.column_ is None,
        )
        return self

    def predict(self, x, exposures):
        if not hasattr(self, "model_"):
            raise ValueError("Fit scale-aware regression before predicting")
        if len(x) != len(exposures):
            raise ValueError("Features and scale inputs must be aligned")
        if isinstance(x, pd.DataFrame) and not x.index.equals(exposures.index):
            raise ValueError("Features and scale inputs must have identical indexes")
        return self.model_.predict(x) * self._scale(exposures)
