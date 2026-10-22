import pickle

import numpy as np
import pandas as pd
import pytest

from graphreduce.distribution_sketch import DistributionSketch


def test_cdf_distinguishes_bags_with_equal_standard_moments():
    a = np.array([-2, -1, -1, 1, 1, 2], dtype=float)
    b = np.array([-2, -np.sqrt(2), 0, 0, np.sqrt(2), 2])
    np.testing.assert_allclose(
        [a.mean(), a.std(), a.min(), a.max()],
        [b.mean(), b.std(), b.min(), b.max()],
        atol=1e-12,
    )
    values = pd.DataFrame({"signal": np.r_[a, b]})
    rows = pd.DataFrame({"__example_id": [1] * 6 + [9] * 6})
    sketch = DistributionSketch(quantiles=(0.5,)).fit(values)
    result = sketch.transform(rows, values, pd.Index([1, 9, 17]))
    assert result.loc[1, "c0__cdf0"] == 0.5
    assert result.loc[9, "c0__cdf0"] == pytest.approx(2 / 3)
    assert result.loc[17].isna().all()


def test_frozen_thresholds_serialization_batch_independence_and_missingness():
    values = pd.DataFrame({"a": [0.0, 1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0, 7.0]})
    rows = pd.DataFrame({"__example_id": [2, 2, 5, 5]})
    sketch = DistributionSketch(quantiles=(0.25, 0.5, 0.75)).fit(values)
    first = sketch.transform(rows, values, pd.Index([2, 5]))
    assert first.loc[5, "c0__coverage"] == 0.5
    restored = pickle.loads(pickle.dumps(sketch))
    pd.testing.assert_frame_equal(
        first, restored.transform(rows, values, pd.Index([2, 5]))
    )
    pd.testing.assert_frame_equal(
        first.loc[[5]],
        restored.transform(rows.iloc[2:], values.iloc[2:], pd.Index([5])),
    )
    before = {k: v.copy() for k, v in restored.cutpoints_.items()}
    restored.transform(rows, values.fillna(999999) * 10000, pd.Index([2, 5]))
    for key in before:
        np.testing.assert_array_equal(before[key], restored.cutpoints_[key])
    with pytest.raises(ValueError, match="schema/order"):
        restored.transform(rows, values[["b", "a"]], pd.Index([2, 5]))
