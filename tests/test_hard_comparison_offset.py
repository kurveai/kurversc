import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from scripts.hard_comparison_offset import HardComparisonOffsetRanker, comparison_gradient, comparison_pairs


@pytest.mark.parametrize("strategy", ["uniform", "mixed_hard"])
def test_pairs_are_unique_bounded_repeatable_and_cohort_local(strategy):
    y = np.tile(np.r_[np.ones(20), np.zeros(80)], 2)
    times = np.repeat(pd.to_datetime(["2020-01-01", "2020-02-01"]), 100)
    p = expit(np.linspace(-4, 4, len(y)))
    pairs = comparison_pairs(y, times, p, comparisons=strategy, maximum=121, seed=3)
    assert pairs.shape == (121, 2) and len(np.unique(pairs, axis=0)) == 121
    assert (y[pairs[:, 0]] == 1).all() and (y[pairs[:, 1]] == 0).all()
    np.testing.assert_array_equal(times[pairs[:, 0]], times[pairs[:, 1]])
    np.testing.assert_array_equal(pairs, comparison_pairs(y, times, p, comparisons=strategy, maximum=121, seed=3))


def test_comparison_gradient_matches_pairwise_objective_derivative():
    z = np.array([-2., 1., 3., 0.])
    pairs = np.array([[0, 1], [0, 2], [3, 2]])
    gradient = comparison_gradient(expit(z), pairs)
    def loss(values):
        return np.logaddexp(0, values[pairs[:, 1]] - values[pairs[:, 0]]).sum()
    for i in range(len(z)):
        delta = np.zeros(len(z)); delta[i] = 1e-5
        derivative = (loss(z + delta) - loss(z - delta)) / 2e-5
        np.testing.assert_allclose(gradient[i], -derivative, rtol=0, atol=1e-9)
    assert abs(gradient.sum()) < 1e-12


def test_mining_uses_nearby_scores_without_held_out_targets():
    y = np.r_[np.ones(30), np.zeros(300)]
    p = expit(np.r_[np.linspace(-.2, .2, 30), np.linspace(-8, 8, 300)])
    t = np.repeat(pd.Timestamp("2020-01-01"), len(y))
    hard = comparison_pairs(y, t, p, comparisons="mixed_hard", maximum=1000, seed=0)
    uniform = comparison_pairs(y, t, p, comparisons="uniform", maximum=1000, seed=0)
    z = np.log(p / (1 - p))
    assert (abs(z[hard[:, 0]] - z[hard[:, 1]]) <= 1).mean() > (abs(z[uniform[:, 0]] - z[uniform[:, 1]]) <= 1).mean() + .25


@pytest.mark.parametrize("utility,comparisons", [("pair_gradient", "uniform"), ("point_residual", "mixed_hard"), ("pair_gradient", "mixed_hard")])
def test_bounded_ranker_serializes_and_rejects_immature_labels(utility, comparisons):
    x = pd.DataFrame({"state":np.arange(40), "constant":np.ones(40), "missing":np.nan})
    y = np.tile([0, 1], 20)
    t = np.repeat(pd.Timestamp("2020-01-01"), len(y))
    kwargs = dict(entities=np.arange(40), timestamps=t, origin="2020-03-01", horizon="30d")
    model = HardComparisonOffsetRanker(utility=utility, comparisons=comparisons, iterations=3, thread_count=1)
    p = expit(np.linspace(-1, 1, len(y)))
    model.fit(x, y, p, **kwargs)
    assert model.columns == ["state"] and model.audit["pairs"] == 400
    np.testing.assert_array_equal(model.predict_values(x, p, strength=1.),
        pickle.loads(pickle.dumps(model)).predict_values(x, p, strength=1.))
    kwargs["origin"] = "2020-01-02"
    with pytest.raises(ValueError, match="matured"):
        model.fit(x, y, p, **kwargs)
    kwargs["origin"] = "2020-03-01"
    x["__event_entity"] = np.arange(40)
    with pytest.raises(ValueError, match="structural"):
        model.fit(x, y, p, **kwargs)
