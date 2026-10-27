import numpy as np
import pandas as pd
import pytest

from kurversc.event_summaries import (
    EventSummaryPlan,
    EventTable,
    _balanced_numeric_pairs,
)


def _tables():
    return {
        "roots": EventTable(pd.DataFrame({"id": [1, 2, 3]}), "id", None, {}),
        "sessions": EventTable(
            pd.DataFrame(
                {
                    "id": [10, 11, 12],
                    "root": [1, 1, 2],
                    "time": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-01"]),
                }
            ),
            "id",
            "time",
            {"root": "roots"},
        ),
        "events": EventTable(
            pd.DataFrame(
                {
                    "session": [10, 10, 10, 11, 12, 12],
                    "dim": [100, 100, 101, 100, 101, 100],
                    "time": pd.to_datetime(
                        [
                            "2020-01-02 00:00:00",
                            "2020-01-01 23:59:59.999999",
                            "2020-01-02 00:00:00.000001",
                            "2020-01-01 23:30:00",
                            "2020-01-01 20:00:00",
                            "2020-01-01 19:00:00",
                        ],
                        format="mixed",
                    ),
                    "value": [2.0, 4.0, 999.0, 888.0, 6.0, 12.0],
                    "flag": [1, 0, 9, 1, 0, 0],
                }
            ),
            None,
            "time",
            {"session": "sessions", "dim": "dimension"},
        ),
        "dimension": EventTable(
            pd.DataFrame(
                {
                    "id": [100, 101, 102],
                    "price": [10.0, 20.0, 30.0],
                    "time": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-01"]),
                }
            ),
            "id",
            "time",
            {},
        ),
    }


def _plan(tables):
    return EventSummaryPlan.fit(tables, "roots", "2020-01-02", max_depth=2)


def test_exact_cutoff_and_intermediate_cutoff_and_duplicate_seeds():
    tables = _tables()
    plan = _plan(tables)
    features = plan.transform(tables, [1, 2, 3, 1], "2020-01-02")
    p = next(i for i, path in enumerate(plan.paths) if path.steps[-1][0] == "events")
    assert features[f"evt_{p:02d}_1h_count"].tolist() == [2.0, 0.0, 0.0, 2.0]
    assert features[f"evt_{p:02d}_6h_count"].tolist() == [2.0, 2.0, 0.0, 2.0]
    v = plan.paths[p].numeric.index("value")
    assert features.iloc[0][f"evt_{p:02d}_1h_p{v}_mean"] == 3.0
    assert np.isnan(features.iloc[2][f"evt_{p:02d}_1h_p{v}_mean"])


def test_vocabulary_learned_before_cutoff_and_frozen():
    tables = _tables()
    plan = _plan(tables)
    p = next(path for path in plan.paths if path.steps[-1][0] == "events")
    assert 9 not in dict(p.categories)["flag"]
    earlier = plan.transform(tables, [1, 2], "2020-01-02", mode="conditional")
    later = plan.transform(tables, [1, 2], "2020-01-04", mode="conditional")
    assert list(earlier) == list(later)


def test_numeric_dimension_is_left_joined_and_point_in_time():
    tables = _tables()
    plan = _plan(tables)
    p = next(i for i, path in enumerate(plan.paths) if path.steps[-1][0] == "events")
    f = plan.transform(tables, [1, 2, 3], "2020-01-02", mode="numeric")
    payloads = list(plan.paths[p].numeric)
    payloads += [
        f"{name}.{c}" for _, name, cols in plan.paths[p].dimensions for c in cols
    ]
    j = payloads.index("dimension.price")
    # Future dimension 101 must not remove the historical fact or expose price.
    assert f.loc[2, f"evt_{p:02d}_30d_count"] == 2.0
    assert f.loc[2, f"evt_{p:02d}_30d_p{j}_support"] == 1.0
    assert f.loc[2, f"evt_{p:02d}_30d_p{j}_sum"] == 10.0


def test_conditional_denominators_are_not_mean_of_means():
    tables = _tables()
    plan = _plan(tables)
    p = next(i for i, path in enumerate(plan.paths) if path.steps[-1][0] == "events")
    cats = dict(plan.paths[p].categories)
    c = list(cats).index("flag")
    k = list(cats["flag"]).index(0)
    v = plan.paths[p].numeric.index("value")
    f = plan.transform(tables, [1, 2], "2020-01-02", mode="conditional")
    prefix = f"evt_{p:02d}_c{c}v{k}"
    assert f.loc[2, prefix + "_support"] == 2
    assert f.loc[2, prefix + f"_p{v}_sum"] == 18
    assert f.loc[2, prefix + f"_p{v}_mean"] == 9


def test_invalid_declared_parent_key_rejected():
    tables = _tables()
    tables["roots"] = EventTable(pd.DataFrame({"id": [1, 1]}), "id", None, {})
    with pytest.raises(ValueError, match="not unique"):
        _plan(tables).transform(tables, [1], "2020-01-02")


def test_nonnumeric_keys_and_quoted_names():
    tables = {
        "a": EventTable(pd.DataFrame({"the key": ["x", "y"]}), "the key", None, {}),
        "b": EventTable(
            pd.DataFrame(
                {
                    "owner": ["x", "x"],
                    "when": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                    'odd"value': [1.0, 3.0],
                }
            ),
            None,
            "when",
            {"owner": "a"},
        ),
    }
    plan = EventSummaryPlan.fit(tables, "a", "2020-01-02")
    result = plan.transform(tables, ["x", "y"], "2020-01-02", mode="numeric")
    assert result.loc["x", "evt_00_30d_p0_mean"] == 2
    assert result.loc["y", "evt_00_30d_count"] == 0


def test_container_payload_is_not_accidentally_expanded():
    tables = _tables()
    tables["events"].frame["list_payload"] = [[1], [2], [3], [4], [5], [6]]
    path = next(p for p in _plan(tables).paths if p.steps[-1][0] == "events")
    assert "list_payload" not in path.numeric
    assert "list_payload" not in dict(path.categories)


def test_distinct_relations_survive_budget_before_duplicate_fk_channels():
    tables = {
        "roots": EventTable(pd.DataFrame({"id": [1, 2]}), "id", None, {}),
        "aaa": EventTable(
            pd.DataFrame(
                {
                    "left": [1],
                    "right": [2],
                    "time": pd.to_datetime(["2020-01-01"]),
                }
            ),
            None,
            "time",
            {"left": "roots", "right": "roots"},
        ),
        "zzz": EventTable(
            pd.DataFrame(
                {
                    "owner": [1],
                    "time": pd.to_datetime(["2020-01-01"]),
                }
            ),
            None,
            "time",
            {"owner": "roots"},
        ),
    }
    plan = EventSummaryPlan.fit(tables, "roots", "2020-01-02", max_paths=2)
    assert [p.steps[-1][0] for p in plan.paths] == ["aaa", "zzz"]
    audit = plan.audit()
    assert not audit["coverage_complete_within_max_depth"]
    assert audit["omitted_event_paths_due_to_budget"] == ((("aaa", "right"),),)


def test_clock_preserves_intraday_distribution_without_future_rows():
    tables = _tables()
    plan = _plan(tables)
    p = next(i for i, path in enumerate(plan.paths) if path.steps[-1][0] == "events")
    frame = plan.transform(tables, [1, 2, 3], "2020-01-02", mode="clock")
    # Exact midnight is in bin zero; a microsecond earlier is in bin three.
    assert frame.loc[1, f"evt_{p:02d}_24h_b0_count"] == 1
    assert frame.loc[1, f"evt_{p:02d}_24h_b3_count"] == 1
    assert frame.loc[1, f"evt_{p:02d}_24h_b3_share"] == 0.5
    assert frame.loc[2, f"evt_{p:02d}_24h_b3_active_days"] == 1
    assert frame.loc[3, f"evt_{p:02d}_24h_b3_share"] == 0


def test_joint_numeric_preserves_row_association_and_availability():
    tables = _tables()
    plan = _plan(tables)
    p = next(i for i, path in enumerate(plan.paths) if path.steps[-1][0] == "events")
    assert plan.paths[p].numeric[:2] == ("flag", "value")
    frame = plan.transform(
        tables, [1, 2, 3, 1], "2020-01-02", mode="joint_numeric", max_numeric_pairs=1
    )
    prefix = f"evt_{p:02d}_30d_p0p1"
    assert frame.iloc[0][prefix + "_support"] == 2
    assert frame.iloc[0][prefix + "_product_mean"] == 1
    assert frame.iloc[0][prefix + "_covariance"] == -0.5
    assert frame.iloc[0][prefix + "_correlation"] == -1
    assert frame.iloc[1][prefix + "_covariance"] == 0
    assert np.isnan(frame.iloc[1][prefix + "_correlation"])
    assert frame.iloc[2][prefix + "_support"] == 0
    pd.testing.assert_series_equal(frame.iloc[0], frame.iloc[3])


def test_numeric_pair_budget_covers_payloads_before_repeating_first_one():
    assert _balanced_numeric_pairs(6, 3) == [(0, 1), (2, 3), (4, 5)]
    assert _balanced_numeric_pairs(4, 2) == [(0, 1), (2, 3)]
    assert _balanced_numeric_pairs(1, 3) == []


@pytest.mark.parametrize(
    "mode", ["subday", "clock", "numeric", "conditional", "joint_numeric"]
)
def test_source_row_order_does_not_change_summaries(mode):
    tables = _tables()
    plan = _plan(tables)
    expected = plan.transform(tables, [2, 1, 3], "2020-01-02", mode=mode)
    shuffled = {
        name: EventTable(
            table.frame.sample(frac=1, random_state=7),
            table.key,
            table.timestamp,
            table.foreign_keys,
        )
        for name, table in tables.items()
    }
    actual = plan.transform(shuffled, [2, 1, 3], "2020-01-02", mode=mode)
    pd.testing.assert_frame_equal(expected, actual)


def test_dimension_only_readout_does_not_include_raw_numeric_payloads():
    tables = _tables()
    plan = _plan(tables)
    f = plan.transform(
        tables, [1, 2], "2020-01-02", mode="numeric", numeric_scope="dimensions"
    )
    p = next(i for i, path in enumerate(plan.paths) if path.steps[-1][0] == "events")
    # Exactly one lifted column (price), distinct from the two raw numerics.
    assert f.loc[1, f"evt_{p:02d}_30d_p0_sum"] == 20
    assert f"evt_{p:02d}_30d_p1_sum" not in f


def test_long_known_numeric_value_survives_a_frozen_database():
    tables = {
        "root": EventTable(pd.DataFrame({"id": [1]}), "id", None, {}),
        "fact": EventTable(
            pd.DataFrame(
                {
                    "root": [1, 1],
                    "time": pd.to_datetime(["2000-01-01", "2001-01-01"]),
                    "value": [2.0, 4.0],
                }
            ),
            None,
            "time",
            {"root": "root"},
        ),
    }
    plan = EventSummaryPlan.fit(tables, "root", "2002-01-01")
    short = plan.transform(
        tables, [1], "2005-01-01", mode="numeric", numeric_scope="raw"
    )
    long = plan.transform(
        tables,
        [1],
        "2005-01-01",
        mode="numeric",
        numeric_scope="raw",
        numeric_history_days=3650,
    )
    assert short.loc[1, "evt_00_730d_count"] == 0
    assert long.loc[1, "evt_00_3650d_p0_latest"] == 4.0
    assert long.loc[1, "evt_00_3650d_p0_mean"] == 3.0
