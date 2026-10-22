import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from kurversc.temporal_policy import _temporal_gain_lower_bound, select_temporal_policy
from scripts.frozen_auc_bootstrap import FrozenAucLayout, frozen_auc_select_temporal_policy, temporal_lower_bound


def test_prepared_auc_matches_sklearn_with_integer_weights_ties_and_zeroes():
    rng = np.random.default_rng(3)
    for n in (32, 401, 1003):
        y = rng.integers(2, size=n)
        for prediction in (rng.random(n), rng.integers(7, size=n) / 7):
            scorer = FrozenAucLayout(y, prediction)
            for _ in range(15):
                w = rng.poisson(1., n)
                if min(w[y == 0].sum(), w[y == 1].sum()) == 0:
                    continue
                assert scorer.score(w) == roc_auc_score(y, prediction, sample_weight=w)
                ids = np.repeat(np.arange(n), w)
                assert scorer.score(w) == roc_auc_score(y[ids], prediction[ids])
    with pytest.raises(ValueError, match="integer"):
        scorer.score(np.full(n, .5))


def test_complete_selector_and_joint_entity_multiplier_bounds_match_exactly():
    rng = np.random.default_rng(4)
    evidence, pairs = [], []
    for origin in range(3):
        y, groups = rng.integers(2, size=100), np.tile(np.arange(25), 4)
        a = rng.random(100)
        b = .5 * a + .5 * y
        pairs.append((y, a, b, groups))
        for seed in range(3):
            for policy, prediction in (("incumbent", a), ("candidate", b)):
                evidence.append(dict(origin=f"2020-0{origin + 1}-01", seed=seed, policy=policy,
                    truth=y, groups=groups, prediction=prediction))
    args = dict(task="classification", bootstrap_repeats=50, minimum_origins=3, minimum_seeds=3)
    assert frozen_auc_select_temporal_policy(evidence, **args) == select_temporal_policy(evidence, **args)
    kw = dict(task="classification", alpha=.001, repeats=100)
    assert temporal_lower_bound(pairs, **kw) == _temporal_gain_lower_bound(pairs, **kw)


def test_regression_policy_unchanged():
    rng = np.random.default_rng(42)
    evidence = []
    for origin in range(3):
        y, groups = rng.normal(size=50), np.tile(np.arange(10), 5)
        for seed in range(3):
            for policy, prediction in (("incumbent", y + .3), ("candidate", y + .1)):
                evidence.append(dict(origin=f"2020-0{origin + 1}-01", seed=seed, policy=policy,
                    truth=y, groups=groups, prediction=prediction))
    args = dict(task="regression", bootstrap_repeats=30, minimum_origins=3, minimum_seeds=3)
    assert frozen_auc_select_temporal_policy(evidence, **args) == select_temporal_policy(evidence, **args)
