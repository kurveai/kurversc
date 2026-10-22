import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc.quantile_regression import JointQuantileRegressor


def test_prediction_replays_and_has_bounded_monotone_quantiles():
    x = pd.DataFrame({"a": np.arange(80) % 13, "b": np.arange(80) % 3})
    model = JointQuantileRegressor(iterations=4, depth=2, thread_count=1).fit(
        x, x.a + x.b
    )
    quantiles = model.predict_quantiles(x, project=True)
    assert quantiles.shape == (80, 5)
    assert np.isfinite(quantiles).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict(x), restored.predict(x))
    np.testing.assert_array_equal(model.predict(x)[:3], model.predict(x.iloc[:3]))
    np.testing.assert_array_equal(model.predict(x, project=True), quantiles[:, 2])


@pytest.mark.parametrize(
    "quantiles", [(0.1, 0.9), (0.5,), (0.5, 0.5), (0.5, 1), (np.nan, 0.5)]
)
def test_invalid_quantiles(quantiles):
    with pytest.raises(ValueError, match="quantiles"):
        JointQuantileRegressor(quantiles=quantiles)


def test_fit_validation_and_constant_targets():
    x = pd.DataFrame({"a": np.arange(20)})
    model = JointQuantileRegressor(iterations=3, depth=2, thread_count=1)
    with pytest.raises(ValueError, match="Fit"):
        model.predict(x)
    with pytest.raises(ValueError, match="Finite"):
        model.fit(x, np.repeat(np.nan, len(x)))
    model.fit(x, np.repeat(7.0, len(x)))
    np.testing.assert_allclose(model.predict(x), 7.0)
