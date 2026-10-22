"""A bounded count-median likelihood challenger, not a production default."""

import numpy as np
from scipy.stats import nbinom

from scripts.overdispersed_count_readout import OverdispersedCountReadout


class CountLikelihoodRegressor:
    """One scalar-tree NB likelihood, decoded for MAE rather than mean error."""

    def __init__(self, *, seed=0, model_params=None):
        self.seed, self.params = seed, dict(model_params or {})

    def fit(self, features, labels, *, categorical):
        self.head = OverdispersedCountReadout(threshold=0, seed=self.seed, model_params=self.params).fit(
            features, labels, categorical=categorical)
        self.audit = {**self.head.audit, "decoder": "negative-binomial conditional median",
                      "target_is_original_count": True, "upper_target_clipping": False}
        return self

    def predict(self, features):
        p = np.asarray(self.head.model.predict_proba(features), float)[:, 1]
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Finite native likelihood probabilities required")
        p = np.clip(p, 1e-12, 1-1e-12)
        result = nbinom.ppf(.5, self.head.dispersion, 1-p)
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("Finite nonnegative conditional count medians required")
        return result
