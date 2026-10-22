import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc.count_regression import CountMedianRegressor
from scripts.poisson_count_median import PoissonCountMedian


def test_packaged_count_readout_reproduces_the_qualified_research_head():
    frame = pd.DataFrame(dict(past=np.arange(90), kind=np.tile(["a", "b", "c"], 30)))
    y = np.tile([0., 1., 2., 4., 8.], 18)
    params = dict(iterations=16, thread_count=1)
    original = PoissonCountMedian(seed=2, model_params=params).fit(frame, y, categorical=[1])
    model = CountMedianRegressor(random_state=2, **params).fit(frame, y, cat_features=[1])
    np.testing.assert_array_equal(original.predict(frame), model.predict(frame))
    np.testing.assert_array_equal(model.predict(frame), pickle.loads(pickle.dumps(model)).predict(frame))
    np.testing.assert_array_equal(original.model.predict(frame, prediction_type="RawFormulaVal"),
                                  model.model_.predict(frame, prediction_type="RawFormulaVal"))
    assert model.model_.feature_names_ == ["past", "kind"]
    assert not model.fit_audit_["enabled_by_default"]
    assert model.predict(frame.iloc[:0]).shape == (0,)


def test_packaged_count_readout_fails_closed_on_invalid_targets_and_budgets():
    frame = pd.DataFrame(dict(past=range(4)))
    for y in ([0., 1., 1., 0.], [0., .5, 1., 2.], [-1., 0., 1., 2.], [0., 1., np.nan, 2.]):
        with pytest.raises(ValueError, match="integer training targets"):
            CountMedianRegressor(iterations=2, thread_count=1).fit(frame, y)
    with pytest.raises(ValueError, match="before predicting"):
        CountMedianRegressor().predict(frame)
    for kwargs in (dict(iterations=0), dict(thread_count=0), dict(learning_rate=np.nan), dict(l2_leaf_reg=-1.)):
        with pytest.raises(ValueError, match="budgets"):
            CountMedianRegressor(**kwargs)
