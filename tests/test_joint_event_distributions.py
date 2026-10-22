import pickle

import duckdb
import numpy as np
import pandas as pd
import pytest

from graphreduce.joint_distribution import JointDistributionSketch
from kurversc.event_summaries import EventTable
from kurversc.joint_event_distributions import JointEventDistributionPlan


def aggregate(sketch, values, groups):
    marginal, joint = sketch.expressions({c: '"' + c + '"' for c in values})
    rows = values.assign(root=groups)
    with duckdb.connect() as con:
        con.register("rows", rows)
        sql = ", ".join(f'{expr} AS "{name}"' for name, expr in (*marginal, *joint))
        result = con.execute(f"SELECT root, {sql} FROM rows GROUP BY root ORDER BY root").df().set_index("root")
    return result[[c for c, _ in marginal]], result[[c for c, _ in joint]]


def test_joint_cdf_separates_equal_marginals_with_different_pairings():
    values = pd.DataFrame({"x": [0., 1., 2., 3.] * 2, "y": [0., 1., 2., 3., 3., 2., 1., 0.]})
    sketch = JointDistributionSketch(max_conditional_pairs=0).fit(values)
    marginal, joint = aggregate(sketch, values, [0] * 4 + [1] * 4)
    np.testing.assert_array_equal(marginal.loc[0], marginal.loc[1])
    assert not np.array_equal(joint.loc[0], joint.loc[1])


def test_conditional_cdf_preserves_category_value_pairing():
    values = pd.DataFrame({"category": ["A", "A", "B", "B"] * 2,
                           "value": [0., 1., 2., 3., 2., 3., 0., 1.]})
    sketch = JointDistributionSketch(max_numeric_pairs=0, max_conditional_pairs=1).fit(values)
    marginal, joint = aggregate(sketch, values, [0] * 4 + [1] * 4)
    np.testing.assert_array_equal(marginal.loc[0], marginal.loc[1])
    assert not np.array_equal(joint.loc[0], joint.loc[1])


def test_missing_values_frozen_thresholds_and_serialization():
    values = pd.DataFrame({"x": [0., 1., 2., np.nan], "y": [0., 1., 2., np.inf]})
    sketch = JointDistributionSketch(max_conditional_pairs=0).fit(values)
    a, b = aggregate(sketch, values, [0] * 4)
    assert a.iloc[0]["n0_coverage"] == .75
    assert a.iloc[0]["n0_support"] == 3
    restored = pickle.loads(pickle.dumps(sketch))
    x, y = aggregate(restored, values, [0] * 4)
    pd.testing.assert_frame_equal(a, x)
    pd.testing.assert_frame_equal(b, y)
    with pytest.raises(ValueError, match="schema"):
        restored.expressions({"x": '"x"'})


def fixture_tables():
    return {
        "roots": EventTable(pd.DataFrame({"id": [1, 2]}), "id", None, {}),
        "facts": EventTable(pd.DataFrame({
            "id": range(10), "root_id": [1] * 5 + [2] * 5,
            "date": pd.to_datetime(["2020-01-01"] * 8 + ["2020-02-01"] * 2),
            "category": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "B"],
            "value": [0., 1., 2., 3., 2., 3., 0., 1., 1000., 2000.],
        }), "id", "date", {"root_id": "roots"}),
    }


def test_asof_queries_duplicate_roots_empty_neighborhoods_and_future_canary():
    tables = fixture_tables()
    plan = JointEventDistributionPlan().fit(tables, "roots", [1, 2], "2020-01-15")
    a, b = plan.transform(tables, [1, 2, 9, 1], "2020-01-15")
    assert len(b.columns)
    pd.testing.assert_series_equal(b.iloc[0], b.iloc[3], check_names=False)
    assert b.iloc[2].isna().all()
    pd.testing.assert_frame_equal(b.iloc[[1]].reset_index(drop=True),
                                  plan.transform(tables, [2], "2020-01-15")[1])
    changed = fixture_tables()
    changed["facts"].frame.loc[changed["facts"].frame.date.ge("2020-01-15"), "value"] = -99999.
    pd.testing.assert_frame_equal(b, plan.transform(changed, [1, 2, 9, 1], "2020-01-15")[1])
    pd.testing.assert_frame_equal(a, pickle.loads(pickle.dumps(plan)).transform(tables, [1, 2, 9, 1], "2020-01-15")[0])
    with pytest.raises(ValueError, match="calibration"):
        plan.transform(tables, [1], "2020-01-14")
