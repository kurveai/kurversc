"""Isolated portable offset correction; no default model changes.

The parent routine supplies all maturity/key/budget checks and bounded training-
only column selection. Its one-tree scaffolding fit is discarded; the actual
240-tree correction uses no bootstrap or randomized split-score perturbation.
"""

import numpy as np
import pandas as pd

from scripts.frozen_offset_ranker import probability_logits
from scripts.hard_comparison_offset import HardComparisonOffsetRanker, comparison_pairs


RECIPES = {
    "portable_gradient": {"utility":"pair_gradient", "comparisons":"uniform"},
    "portable_residual": {"utility":"point_residual", "comparisons":"uniform"},
    "portable_consensus": {"utility":"consensus", "comparisons":"uniform"},
}


class PortableComparisonOffsetRanker(HardComparisonOffsetRanker):
    def __init__(self, *, utility, comparisons="uniform", **kwargs):
        if utility not in {"pair_gradient", "point_residual", "consensus"} or comparisons != "uniform":
            raise ValueError("Only predeclared portable correction recipes allowed")
        kwargs["thread_count"] = 1
        super().__init__(utility="pair_gradient" if utility == "consensus" else utility,
                         comparisons=comparisons, **kwargs)
        self.consensus = utility == "consensus"

    def fit(self, features, labels, baseline, **fit_args):
        from catboost import CatBoostRanker, Pool

        # Match the model's actual precision before ranking near-tied values.
        x = pd.DataFrame(features).copy()
        for column in x.select_dtypes(include=["number"]):
            x[column] = x[column].astype("float32")
        if self.consensus:
            self.members = [PortableComparisonOffsetRanker(utility=u, seed=self.seed,
                maximum_columns=self.maximum_columns, iterations=self.iterations)
                .fit(x, labels, baseline, **fit_args) for u in ("point_residual", "pair_gradient")]
            self.audit = dict(strategy="fixed equal portable objective consensus", seed=self.seed,
                members=[m.audit for m in self.members], training_only_selection=True,
                bootstrap_type="No", random_strength=0., thread_count=1,
                structural_keys_are_predictors=False)
            return self
        iterations = self.iterations
        self.iterations = 1
        try:
            super().fit(x, labels, baseline, **fit_args)
        finally:
            self.iterations = iterations
        times = pd.DatetimeIndex(pd.to_datetime(fit_args["timestamps"]))
        order = np.argsort(times.asi8, kind="stable")
        y = np.asarray(labels, dtype=float)
        pairs = comparison_pairs(y[order], times[order], np.asarray(baseline)[order],
            comparisons="uniform", maximum=100_000, seed=self.seed + 61010)
        numeric = x[self.columns].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        pool = Pool(numeric.iloc[order].to_numpy(dtype="float32"), y[order], pairs=pairs,
            group_id=pd.factorize(times[order], sort=False)[0],
            baseline=probability_logits(baseline)[order].astype("float32"))
        self.model = CatBoostRanker(loss_function="PairLogit", iterations=iterations,
            depth=3, learning_rate=.03, l2_leaf_reg=30., random_seed=self.seed,
            thread_count=1, bootstrap_type="No", random_strength=0., verbose=False,
            allow_writing_files=False).fit(pool)
        self.audit.update(iterations=iterations, discarded_scaffolding_trees=1,
            bootstrap_type="No", random_strength=0., thread_count=1,
            input_precision="float32 before utility/Pool", fixed_baseline_precision="float32")
        return self

    def predict_values(self, features, baseline, *, strength):
        if not self.consensus:
            return super().predict_values(features, baseline, strength=strength)
        if strength != 1.:
            raise ValueError("Consensus has one predeclared full strength")
        return np.mean([m.predict_values(features, baseline, strength=1.) for m in self.members], axis=0)
