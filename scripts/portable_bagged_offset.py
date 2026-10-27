"""Fixed three-comparison-sample ensemble of portable gradient corrections."""

import numpy as np

from scripts.portable_offset_readout import PortableComparisonOffsetRanker


RECIPES = {"portable_gradient_bag3":{"utility":"pair_gradient", "comparisons":"uniform"}}


class PortableBaggedOffsetRanker:
    def __init__(self, *, seed=0, utility="pair_gradient", comparisons="uniform"):
        if utility != "pair_gradient" or comparisons != "uniform":
            raise ValueError("Only one predeclared equal-weight portable bag supported")
        self.seed = seed

    def fit(self, features, labels, baseline, **fit_args):
        self.models = [PortableComparisonOffsetRanker(seed=s, utility="pair_gradient")
            .fit(features, labels, baseline, **fit_args) for s in range(3*self.seed, 3*self.seed+3)]
        self.audit = dict(seed=self.seed, strategy="equal portable gradient correction probabilities",
            member_seeds=[m.seed for m in self.models], members=[m.audit for m in self.models],
            chosen_seeds_or_weights_from_outcomes=False, structural_keys_are_predictors=False)
        return self

    def predict_values(self, features, baseline, *, strength):
        if strength != 1.:
            raise ValueError("One fixed full-strength ensemble only")
        return np.mean([m.predict_values(features, baseline, strength=1.) for m in self.models], axis=0)


class PortableBaggedRecipeReadout:
    def __init__(self, *, recipe, seed=0):
        if recipe not in RECIPES or seed not in (0, 1, 2):
            raise ValueError("Only declared bagged recipe and deployment seeds allowed")
        self.model = PortableBaggedOffsetRanker(seed=seed, **RECIPES[recipe])

    def fit(self, features, labels, baseline, **fit_args):
        self.model.fit(features, labels, baseline, **fit_args)
        self.audit = self.model.audit
        return self

    def predict_values(self, features, baseline):
        return self.model.predict_values(features, baseline, strength=1.)
