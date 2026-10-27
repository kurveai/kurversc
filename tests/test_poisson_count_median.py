import pickle

import numpy as np
import pandas as pd

from scripts.poisson_count_median import PoissonCountMedian


def test_poisson_conditional_count_median_is_bounded_and_serializes():
    frame = pd.DataFrame(dict(past=np.arange(160)))
    labels = np.repeat([0, 1, 5, 20], 40)
    model = PoissonCountMedian(model_params=dict(iterations=20, thread_count=1)).fit(frame, labels, categorical=[])
    prediction = model.predict(frame)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    assert np.unique(prediction).size > 1
    np.testing.assert_array_equal(prediction, np.floor(prediction))
    np.testing.assert_array_equal(prediction, pickle.loads(pickle.dumps(model)).predict(frame))
    assert not model.audit["marginal_variance_used_as_conditional_dispersion"]
