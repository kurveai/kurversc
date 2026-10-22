import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.count_likelihood_regressor import CountLikelihoodRegressor


def test_count_median_is_finite_integral_and_pickle_exact():
    frame = pd.DataFrame(dict(history=np.arange(180), label_free=np.arange(180) % 7))
    y = np.repeat([0, 1, 4, 12, 30, 50], 30)
    model = CountLikelihoodRegressor(model_params=dict(iterations=12, thread_count=1)).fit(frame, y, categorical=[])
    prediction = model.predict(frame)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    np.testing.assert_array_equal(prediction, np.floor(prediction))
    np.testing.assert_array_equal(prediction, pickle.loads(pickle.dumps(model)).predict(frame))
    assert not model.audit["upper_target_clipping"]
    assert model.audit["output_dimensions"] == 1


def test_nonintegral_count_regression_is_unavailable():
    frame = pd.DataFrame(dict(past=np.arange(40)))
    with pytest.raises(ValueError, match="integral"):
        CountLikelihoodRegressor().fit(frame, np.arange(40) / 3, categorical=[])
