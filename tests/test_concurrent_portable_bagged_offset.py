import numpy as np
import pandas as pd

from scripts.concurrent_portable_bagged_offset import ConcurrentPortableBaggedOffsetRanker
from scripts.portable_bagged_offset import PortableBaggedOffsetRanker


def test_member_concurrency_preserves_exact_predictions_and_order():
    x = pd.DataFrame({"state":np.arange(40), "second":np.tile([0., 1., 2., 3.], 10)})
    y, p = np.tile([0, 1], 20), np.linspace(.1, .7, 40)
    args = dict(entities=np.arange(40), timestamps=np.repeat(pd.Timestamp("2020-01-01"), 40),
                origin="2020-03-01", horizon="30d")
    serial = PortableBaggedOffsetRanker(seed=1).fit(x, y, p, **args)
    parallel = ConcurrentPortableBaggedOffsetRanker(seed=1).fit(x, y, p, **args)
    assert [m.seed for m in parallel.models] == [3, 4, 5]
    np.testing.assert_array_equal(serial.predict_values(x, p, strength=1.), parallel.predict_values(x, p, strength=1.))
    for a, b in zip(serial.models, parallel.models):
        assert a.columns == b.columns
        np.testing.assert_array_equal(a.predict_values(x, p, strength=1.), b.predict_values(x, p, strength=1.))
