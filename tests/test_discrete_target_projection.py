import numpy as np
import pandas as pd
import pytest

from scripts.discrete_target_projection import TargetDomainProjection


def fit(labels):
    return TargetDomainProjection().fit(labels,["2020-01-01"]*len(labels),
        origin="2020-04-01",horizon="30D")


def test_integer_support_and_no_upper_clipping():
    head = fit([0,1,2,0,1])
    assert head.integer_counts
    np.testing.assert_array_equal(head.predict_values([-1,.2,1.2,999.9],recipe="integer_counts"),[0,0,1,1000])
    assert head.audit["upper_bound"] is None


def test_continuous_and_negative_support_not_forced_to_counts():
    head = fit([0,.5,1.25])
    assert head.nonnegative and not head.integer_counts
    with pytest.raises(ValueError):
        head.predict_values([.3],recipe="integer_counts")
    head = fit([-1,0,1])
    assert not head.nonnegative and not head.integer_counts
    with pytest.raises(ValueError):
        head.predict_values([.3],recipe="nonnegative")


def test_incomplete_and_future_labels_rejected():
    for labels in ([0,np.nan,1],[]):
        with pytest.raises(ValueError):
            fit(labels)
    with pytest.raises(ValueError):
        TargetDomainProjection().fit([0,1,2],["2020-03-20"]*3,
            origin="2020-04-01",horizon="30D")


def test_lower_bound_clipping_never_worsens_nonnegative_mae():
    head = fit([0,1,2])
    y = np.array([0,1,2,0])
    raw = np.array([-2,-1,5,.1])
    assert np.mean(abs(y-head.predict_values(raw,recipe="nonnegative"))) <= np.mean(abs(y-raw))
