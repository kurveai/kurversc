import pickle

import numpy as np
import pandas as pd

from kurversc.ordered_regression import OrderedTargetRegressor


def test_ordered_threshold_probabilities_serialization_and_batch_replay():
    x = pd.DataFrame({"a": np.arange(90) % 17, "b": np.arange(90) % 3})
    y = x.a + x.b * 2
    model = OrderedTargetRegressor(bins=8, iterations=4, depth=2, thread_count=1).fit(
        x, y
    )
    mass = model.predict_distribution(x)
    assert mass.shape == (90, len(model.classes_))
    assert (mass >= 0).all()
    np.testing.assert_allclose(mass.sum(axis=1), 1)
    assert (np.diff(mass.cumsum(axis=1), axis=1) >= 0).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict(x), restored.predict(x))
    np.testing.assert_array_equal(model.predict(x)[:3], model.predict(x.iloc[:3]))
    before = model.cutpoints_.copy()
    model.predict(x.assign(a=100000))
    np.testing.assert_array_equal(before, model.cutpoints_)


def test_binary_and_constant_target_special_cases():
    x = pd.DataFrame({"a": np.arange(20)})
    for y in ((x.a > 10).astype(float), np.ones(20) * 7):
        model = OrderedTargetRegressor(iterations=3, depth=2, thread_count=1).fit(x, y)
        assert np.isfinite(model.predict(x)).all()
        np.testing.assert_allclose(model.predict_distribution(x).sum(axis=1), 1)


def test_explicit_regularization_retains_existing_default():
    x = pd.DataFrame({"a": np.arange(20)})
    assert OrderedTargetRegressor().l2_leaf_reg == 3.0
    model = OrderedTargetRegressor(
        iterations=3, depth=2, thread_count=1, l2_leaf_reg=10
    ).fit(x, x.a)
    assert model.model_.get_params()["l2_leaf_reg"] == 10
