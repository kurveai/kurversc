import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.special import betainc
from scipy.stats import nbinom

from scripts.overdispersed_count_readout import OverdispersedCountReadout, training_dispersion


def test_count_likelihood_moments_and_probability_convention():
    values = np.array([0] * 50 + [1] * 20 + [2] * 10 + [10] * 20)
    expected = values.mean() ** 2 / (values.var() - values.mean())
    assert training_dispersion(values) == pytest.approx(np.clip(expected, .1, 100))
    for shape in (.1, .8, 4.):
        for probability in (.001, .1, .8):
            assert betainc(2, shape, probability) == pytest.approx(nbinom.sf(1, shape, 1 - probability))
    with pytest.raises(ValueError):
        training_dispersion([0, 1, -1])


def test_soft_label_count_head_fits_repeats_and_serializes():
    counts = np.tile([0, 0, 0, 1, 1, 2, 4, 8], 10)
    features = pd.DataFrame(dict(x=np.arange(len(counts)) % 8, cat=["a", "b"] * 40))
    kwargs = dict(threshold=1, model_params=dict(iterations=5, thread_count=1))
    model = OverdispersedCountReadout(**kwargs).fit(features, counts, categorical=[1])
    predictions = model.predict_proba(features)
    assert predictions.shape == (len(counts), 2) and np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions.sum(axis=1), 1)
    np.testing.assert_array_equal(predictions, pickle.loads(pickle.dumps(model)).predict_proba(features))
    repeat = OverdispersedCountReadout(**kwargs).fit(features, counts, categorical=[1])
    np.testing.assert_array_equal(predictions, repeat.predict_proba(features))
