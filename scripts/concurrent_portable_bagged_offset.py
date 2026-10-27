"""Execution-only concurrency for the same three fixed portable fit members."""

from concurrent.futures import ThreadPoolExecutor

from scripts.portable_bagged_offset import PortableBaggedOffsetRanker
from scripts.portable_offset_readout import PortableComparisonOffsetRanker


class ConcurrentPortableBaggedOffsetRanker(PortableBaggedOffsetRanker):
    def fit(self, features, labels, baseline, **fit_args):
        seeds = list(range(3 * self.seed, 3 * self.seed + 3))

        def fit_one(seed):
            return PortableComparisonOffsetRanker(seed=seed, utility="pair_gradient").fit(
                features, labels, baseline, **fit_args)

        with ThreadPoolExecutor(max_workers=3) as pool:
            self.models = list(pool.map(fit_one, seeds))
        self.audit = dict(seed=self.seed, strategy="equal portable gradient correction probabilities",
            member_seeds=[m.seed for m in self.models], members=[m.audit for m in self.models],
            chosen_seeds_or_weights_from_outcomes=False, structural_keys_are_predictors=False,
            execution_member_workers=3, model_threads_per_member=1)
        return self
