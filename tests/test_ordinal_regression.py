import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc.ordinal_regression import OrdinalDistributionRegressor


def test_distribution_statistics_are_distinct_and_training_only():
    x = pd.DataFrame({"a": np.arange(64) % 7, "b": np.arange(64) % 3})
    y = (np.arange(64) % 5 == 0).astype(float)
    model = OrdinalDistributionRegressor(iterations=3, bins=16, thread_count=1).fit(
        x, y
    )
    probability = model.predict_distribution(x)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert set(model.predict(x)).issubset({0.0, 1.0})
    np.testing.assert_allclose(model.predict(x, statistic="mean"), probability[:, 1])
    restored = pickle.loads(pickle.dumps(model))
    for mode in ("median", "mean"):
        np.testing.assert_array_equal(
            model.predict(x, statistic=mode), restored.predict(x, statistic=mode)
        )
        np.testing.assert_array_equal(
            model.predict(x.iloc[:2], statistic=mode),
            model.predict(x, statistic=mode)[:2],
        )
    before = model.cutpoints_.copy()
    model.predict(x.assign(a=10000))
    np.testing.assert_array_equal(before, model.cutpoints_)


def test_constant_target_fallback_and_invalid_targets():
    x = pd.DataFrame({"a": [1, 2, 3]})
    model = OrdinalDistributionRegressor().fit(x, [7.0, 7.0, 7.0])
    np.testing.assert_array_equal(model.predict(x), [7, 7, 7])
    with pytest.raises(ValueError, match="Finite"):
        model.fit(x, [0, np.nan, 1])
    with pytest.raises(ValueError, match="Statistic"):
        model.predict(x, statistic="mode")
