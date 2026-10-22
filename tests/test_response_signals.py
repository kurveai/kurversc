import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc.event_summaries import EventTable
from kurversc.response_signals import ResponseSignalPlan, mature_own_history


def tables():
    return {
        "roots": EventTable(pd.DataFrame({"key": [1, 2, 3]}), "key", None, {}),
        "dimensions": EventTable(pd.DataFrame({"key": [10, 20], "kind": ["A", "B"], "amount": [10., 100.]}), "key", None, {}),
        "facts": EventTable(pd.DataFrame({
            "key": [0, 1, 2, 3, 4], "owner": [1, 1, 2, 2, 1],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03", "2020-02-01"]),
            "dimension": [10, 20, 10, 20, 10], "binary_payload": [0., 1., 1., 0., 1.],
            "measure": [10., 100., 10., 100., 99999.],
        }), "key", "date", {"owner": "roots", "dimension": "dimensions"}),
    }


def test_response_context_separates_identical_marginals():
    source = tables()
    plan = ResponseSignalPlan().fit(source, "roots", [1, 2], "2020-01-05", label_horizon="4d")
    result = plan.transform(source, [1, 2], "2020-01-05")
    np.testing.assert_array_equal(result["context_marginal"].iloc[0], result["context_marginal"].iloc[1])
    assert not result["response_context"].iloc[0].equals(result["response_context"].iloc[1])
    assert plan.plans_[0]["responses"] == ["binary_payload"]


def test_strict_future_cutoff_empty_roots_duplicates_and_serialization():
    source = tables()
    plan = ResponseSignalPlan().fit(source, "roots", [1, 2], "2020-01-03", label_horizon="4d")
    a = plan.transform(source, [1, 2, 3, 1], "2020-01-05")
    for result in a.values():
        pd.testing.assert_series_equal(result.iloc[0], result.iloc[3], check_names=False)
    assert a["response_momentum"].iloc[2].filter(like="_support").eq(0).all()
    changed = tables()
    changed["facts"].frame.loc[changed["facts"].frame.date.ge("2020-01-05"), ["binary_payload", "measure"]] = [0., -99999.]
    b = plan.transform(changed, [1, 2, 3, 1], "2020-01-05")
    for arm in a:
        pd.testing.assert_frame_equal(a[arm], b[arm])
        pd.testing.assert_frame_equal(a[arm], pickle.loads(pickle.dumps(plan)).transform(source, [1, 2, 3, 1], "2020-01-05")[arm])
    exact = plan.transform(source, [1, 2], "2020-01-03")
    assert exact["response_momentum"].filter(like="w0_support").iloc[:, 0].tolist() == [1., 1.]
    with pytest.raises(ValueError, match="calibration"):
        plan.transform(source, [1], "2020-01-02")


def test_own_history_excludes_immature_same_row_and_phase_future_labels():
    history = pd.DataFrame({"entity": [1, 1, 1, 2],
        "time": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-09", "2020-01-01"]), "y": [1., 0., 999., 0.]})
    kwargs = dict(entity="entity", timestamp="time", target="y", label_horizon="4d")
    a = mature_own_history(history, [1, 2, 3, 1], "2020-01-09", **kwargs)
    assert a.own_history_count.tolist() == [1., 1., 0., 1.]
    assert a.own_history_latest.iloc[0] == 1.
    assert a.own_history_mean.iloc[1] == 0.
    b = mature_own_history(history, [1, 2, 3, 1], "2020-01-12", phase_cutoff="2020-01-09", **kwargs)
    assert b.own_history_count.tolist() == [1., 1., 0., 1.]
    changed = history.copy()
    changed.loc[changed.time.ge("2020-01-05"), "y"] = -9999.
    pd.testing.assert_frame_equal(a, mature_own_history(changed, [1, 2, 3, 1], "2020-01-09", **kwargs))
    empty = mature_own_history(history, [1, 2], "2020-01-05", **kwargs)
    assert empty.own_history_count.eq(0).all()


def test_nonunique_dimension_key_rejected():
    source = tables()
    frame = source["dimensions"].frame
    source["dimensions"] = EventTable(pd.concat([frame, frame.iloc[[0]]]), "key", None, {})
    with pytest.raises(ValueError, match="Non-unique"):
        ResponseSignalPlan().fit(source, "roots", [1], "2020-01-05", label_horizon="4d")


def test_boolean_labels_and_boolean_context_values():
    history = pd.DataFrame({"entity": [1, 2], "time": pd.to_datetime(["2020-01-01"]*2), "y": [True, False]})
    result = mature_own_history(history, [1, 2], "2020-01-06", entity="entity", timestamp="time", target="y", label_horizon="4d")
    assert result.own_history_mean.tolist() == [1., 0.]
    source = tables()
    frame = source["dimensions"].frame.copy()
    frame["boolean_context"] = [False, True]
    source["dimensions"] = EventTable(frame, "key", None, {})
    plan = ResponseSignalPlan().fit(source, "roots", [1, 2], "2020-01-05", label_horizon="4d")
    c = next(i for i, context in enumerate(plan.plans_[0]["contexts"]) if context['source'] == 'dimensions.boolean_context')
    output = plan.transform(source, [1, 2], "2020-01-05")
    assert output['context_marginal'][f'response_p0_r0_all_c{c}_b0_support'].tolist() == [1., 1.]
    assert output['context_marginal'][f'response_p0_r0_all_c{c}_b1_support'].tolist() == [1., 1.]
