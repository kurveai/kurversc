import pickle

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from kurversc import LearnedMessageFamily, Table, Relationship
from graphreduce.learned_messages import extract_path, pool_messages


def fixture():
    roots = Table(pd.DataFrame({"id": range(6)}), key="id", name="roots", timeless=True)
    child = Table(
        pd.DataFrame(
            {
                "row": range(18),
                "parent": np.repeat(range(6), 3),
                "date": pd.to_datetime(["2000-01-01"] * 12 + ["2040-01-01"] * 6),
                "signal": np.arange(18) ** 2,
                "category": ["a", "b", "c"] * 6,
            }
        ),
        key="row",
        name="child",
        date="date",
    )
    tables = {"roots": roots, "child": child}
    edges = (Relationship("roots", "child", "id", "parent"),)
    train = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "time": pd.to_datetime(
                ["2001-01-01", "2001-01-01", "2003-01-01", "2003-01-01"]
            ),
            "y": [0.0, 1.0, 0.2, 0.8],
            "ready": pd.to_datetime(["2002-01-01"] * 2 + ["2004-01-01"] * 2),
        }
    )
    query = pd.DataFrame(
        {
            "id": range(6),
            "time": pd.to_datetime(
                [
                    "2002-01-01",
                    "2003-01-01",
                    "2003-01-01",
                    "2005-01-01",
                    "2005-01-01",
                    "2005-01-01",
                ]
            ),
        }
    )
    return tables, edges, train, query


def fit_family(training=None):
    tables, edges, train, query = fixture()
    family = LearnedMessageFamily(iterations=8, max_neighbors=2, thread_count=1)
    result = family.fit_transform(
        train if training is None else training,
        query,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="id",
        timestamp="time",
        target="y",
        available_at="ready",
    )
    return family, result


def test_future_labels_and_query_targets_cannot_change_earlier_messages():
    family, (raw, messages) = fit_family()
    tables, edges, train, query = fixture()
    train.loc[2:, "y"] = [99.0, -99.0]
    _, (_, changed) = fit_family(train)
    assert_frame_equal(messages.iloc[:3], changed.iloc[:3])
    assert messages.loc[0, "learned_p0__available"] == 0  # equality unavailable
    assert messages.loc[1, "learned_p0__available"] == 1
    assert messages.loc[4, "learned_p0__available"] == 0  # future neighbors absent
    query["y"] = 999
    replay = pickle.loads(pickle.dumps(family)).transform(
        query, tables=tables, relationships=edges
    )
    assert_frame_equal(messages, replay[1])
    assert_frame_equal(raw, replay[0])
    assert all(
        pd.Timestamp(a["latest_label_available"]) < pd.Timestamp(a["cutoff"])
        for a in family.audit
    )


def test_bounded_extraction_is_order_independent_and_excludes_keys():
    tables, edges, _, query = fixture()
    kwargs = dict(
        root_name="roots",
        entity_key="id",
        timestamp="time",
        path=("child",),
        max_neighbors=2,
    )
    rows = extract_path(query, tables, edges, **kwargs)
    assert rows.groupby("__example_id").size().max() == 2
    assert set(rows) == {"__example_id", "signal", "category", "__age_days"}
    from dataclasses import replace

    shuffled = {
        name: replace(table, source=table.source.sample(frac=1, random_state=7))
        for name, table in tables.items()
    }
    assert_frame_equal(rows, extract_path(query, shuffled, edges, **kwargs))
    values = pd.DataFrame({"score": np.ones(len(rows))})
    pooled = pool_messages(rows, values, query.index)
    assert pooled.loc[4, "neighbor_count"] == 0
    assert pd.isna(pooled.loc[4, "score__mean"])


def test_bad_label_availability_rejected():
    _, _, train, _ = fixture()
    train.loc[0, "ready"] = pd.Timestamp("1990-01-01")
    with pytest.raises(ValueError, match="availability"):
        fit_family(train)


def test_final_scorer_includes_last_matured_labels_without_boundary_rounding():
    family, _ = fit_family()
    final = max(family.audit, key=lambda a: pd.Timestamp(a["cutoff"]))
    assert pd.Timestamp(final["latest_label_available"]) == pd.Timestamp("2004-01-01")
    assert final["training_roots"] == 4


def test_new_timestamp_uses_frozen_scorer_and_binary_probabilities():
    tables, edges, train, query = fixture()
    train["y"] = [0, 1, 0, 1]
    family = LearnedMessageFamily(iterations=8, thread_count=1)
    family.fit_transform(
        train,
        query,
        tables=tables,
        relationships=edges,
        root_name="roots",
        entity_key="id",
        timestamp="time",
        target="y",
        available_at="ready",
        task="classification",
    )
    count = len(family.audit)
    query["time"] = pd.Timestamp("2006-01-01")
    _, messages = family.transform(query, tables=tables, relationships=edges)
    values = messages["learned_p0__score__mean"].dropna()
    assert values.between(0, 1).all()
    assert len(family.audit) == count
    table, edge = family.as_table(messages)
    assert table.cutoff_boundary == "exact"
    assert edge.parent == "roots"


def test_exact_snapshot_cutoff_materialization():
    from kurversc.core import _table_workspace, _materialize_at_cutoff
    from kurversc import GraphConfig, Labels

    roots = Table(pd.DataFrame({"id": [1, 2]}), key="id", name="roots", timeless=True)
    snap = Table(
        pd.DataFrame(
            {
                "id": [1, 1, 1, 2],
                "date": pd.to_datetime(
                    ["2001-01-01", "2002-01-01", "2003-01-01", "2002-01-01"]
                ),
                "score": [100.0, 2.0, 999.0, 3.0],
            }
        ),
        name="messages",
        key="id",
        date="date",
        cutoff_boundary="exact",
        auto_base_predicate_max=0,
    )
    labels = pd.DataFrame(
        {"id": [1, 2], "timestamp": pd.Timestamp("2002-01-01"), "target": [0.0, 1.0]}
    )
    spec = Labels(
        labels, target="target", key="id", timestamp="timestamp", target_history=False
    )
    tables = {"roots": roots, "messages": snap}
    with _table_workspace(
        None,
        tables,
        sample_rows=100,
        random_state=0,
        search=False,
        duckdb_memory_limit="1GB",
    ) as workspace:
        frame = _materialize_at_cutoff(
            workspace,
            tables,
            [Relationship("roots", "messages", "id", "id")],
            "roots",
            GraphConfig(auto_annotate_features=False),
            labels,
            spec,
            cut_date=pd.Timestamp("2002-01-01"),
            compute_period_days=3650,
            verbose=False,
        )
    score = next(c for c in frame if "score_avg" in c)
    np.testing.assert_allclose(frame[score], [2.0, 3.0])


def test_message_snapshots_integrate_with_public_fit_and_predict():
    import kurversc

    family, _ = fit_family()
    tables, edges, _, query = fixture()
    early = query.iloc[:4].copy()
    early["time"] = pd.Timestamp("2003-01-01")
    late = early.copy()
    late["time"] = pd.Timestamp("2005-01-01")
    labels = pd.concat([early, late], ignore_index=True)
    labels["y"] = [0.0, 1.0, 0.2, 0.8] * 2
    labels["split"] = ["train"] * 4 + ["validation"] * 4
    _, snapshots = family.transform(labels, tables=tables, relationships=edges)
    message_table, message_edge = family.as_table(snapshots)
    result = kurversc.fit(
        tables["roots"],
        kurversc.Labels(
            labels,
            key="id",
            timestamp="time",
            target="y",
            split="split",
            target_history=False,
        ),
        tables=[message_table],
        relationships=[message_edge],
        task="regression",
        graph_configs=[
            kurversc.GraphConfig(
                auto_annotate_features=False, include_target_history=False
            )
        ],
        model_params={"iterations": 8, "depth": 2, "thread_count": 1},
        rerank_top_k=0,
        confirmation_top_k=0,
        sample_rows=100,
        duckdb_memory_limit="1GB",
        continue_on_error=False,
    )
    future = early.copy()
    future["time"] = pd.Timestamp("2006-01-01")
    _, future_snapshots = family.transform(future, tables=tables, relationships=edges)
    future_table, future_edge = family.as_table(future_snapshots)
    prediction = kurversc.predict(
        result,
        tables["roots"],
        kurversc.Labels(future, key="id", timestamp="time"),
        tables=[future_table],
        relationships=[future_edge],
        duckdb_memory_limit="1GB",
    )
    assert len(prediction) == len(future)
    assert prediction["prediction"].notna().all()
