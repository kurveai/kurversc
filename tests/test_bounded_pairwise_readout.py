import numpy as np
import pandas as pd
import pytest

from scripts.bounded_pairwise_readout import bounded_positive_pairs, BoundedPairwiseReadout


def test_bounded_deterministic_positive_negative_pairs():
    y = np.array([0] * 1000 + [1] * 100)
    a = bounded_positive_pairs(y, maximum=300, seed=42)
    np.testing.assert_array_equal(a, bounded_positive_pairs(y, maximum=300, seed=42))
    assert a.shape == (300, 2)
    assert len(np.unique(a, axis=0)) == 300
    assert (y[a[:, 0]] == 1).all() and (y[a[:, 1]] == 0).all()
    assert bounded_positive_pairs([1, 0, 1, 0], maximum=10).shape == (4, 2)
    with pytest.raises(ValueError):
        bounded_positive_pairs([0, 0])


def test_small_actual_ranker_calibrates_and_repeats_without_holdout():
    x = pd.DataFrame({"value": np.arange(40), "nominal": ["a", "b"] * 20})
    y = np.array([0] * 30 + [1] * 10)
    model = BoundedPairwiseReadout(seed=0, model_params={"iterations": 3, "depth": 3,
        "learning_rate": .05, "l2_leaf_reg": 30, "thread_count": 1,
        "auto_class_weights": "SqrtBalanced"}).fit(x, y, categorical=[1])
    a = model.predict_proba(x)
    np.testing.assert_array_equal(a, model.predict_proba(x))
    assert a.shape == (40, 2) and np.isfinite(a).all()
    assert model.audit["pairs"] == 300 and model.audit["training_rows"] == 40
