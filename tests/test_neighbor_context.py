from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from graphreduce.learned_messages import extract_path
from kurversc import Table, Relationship, LearnedMessageFamily


def schema():
    tables = {
        "entities": Table(pd.DataFrame({"id": [1, 2]}), key="id", timeless=True),
        "neighbors": Table(
            pd.DataFrame(
                {
                    "row": [10, 11, 12],
                    "parent": [1, 1, 2],
                    "signal": [0.2, 0.8, 0.5],
                    "date": pd.Timestamp("2000-01-01"),
                }
            ),
            key="row",
            date="date",
        ),
        "history": Table(
            pd.DataFrame(
                {
                    "event": [100, 101, 102, 103],
                    "neighbor": [10, 10, 11, 12],
                    "value": [2.0, 9999.0, 8.0, 5.0],
                    "date": pd.to_datetime(
                        ["2001-01-01", "2040-01-01", "2001-01-01", "2001-01-01"]
                    ),
                }
            ),
            key="event",
            date="date",
        ),
    }
    edges = [
        Relationship("entities", "neighbors", "id", "parent"),
        Relationship("neighbors", "history", "row", "neighbor"),
    ]
    examples = pd.DataFrame(
        {
            "id": [1, 2, 1],
            "time": pd.to_datetime(["2003-01-01", "2003-01-01", "2004-01-01"]),
        }
    )
    return tables, edges, examples


def extract(tables, edges, examples, depth):
    return extract_path(
        examples,
        tables,
        edges,
        root_name="entities",
        entity_key="id",
        timestamp="time",
        path=("neighbors",),
        context_depth=depth,
        context_max_columns=2,
        max_neighbors=128,
        memory_limit="1GB",
    )


def test_native_graphreduce_enriches_before_pooling_and_excludes_future_rows():
    tables, edges, examples = schema()
    plain = extract(tables, edges, examples, 0)
    enriched = extract(tables, edges, examples, 1)
    assert len(enriched) == len(plain)
    assert_frame_equal(plain, enriched[list(plain)])
    value = next(c for c in enriched if c.endswith("value_avg"))
    np.testing.assert_allclose(enriched[value], [2.0, 8.0, 5.0, 2.0, 8.0])
    assert not any(c.startswith("__neighbor_key") for c in enriched)
    assert not any(
        c.endswith(("event_avg", "event_sum", "event_min", "event_max"))
        for c in enriched
    )
    changed = tables["history"].source.copy()
    changed.loc[1, "value"] = -99999
    tables["history"] = replace(tables["history"], source=changed)
    assert_frame_equal(enriched, extract(tables, edges, examples, 1))


def test_enriched_family_replays_without_treating_generated_columns_as_raw_payload():
    tables, edges, examples = schema()
    training = examples.copy()
    training["y"] = [0.2, 0.8, 0.3]
    training["ready"] = training.time + pd.Timedelta(days=1)
    family = LearnedMessageFamily(
        max_depth=1,
        neighbor_context_depth=1,
        iterations=8,
        thread_count=1,
        memory_limit="1GB",
    )
    first = family.fit_transform(
        training,
        examples,
        tables=tables,
        relationships=edges,
        root_name="entities",
        entity_key="id",
        timestamp="time",
        target="y",
        available_at="ready",
    )
    second = family.transform(examples, tables=tables, relationships=edges)
    for a, b in zip(first, second):
        assert_frame_equal(a, b)
    assert any("neighbor_context" in c for c in first[0])


def test_sibling_context_relations_do_not_multiply_each_others_rows():
    tables, edges, examples = schema()
    tables["measurements"] = Table(
        pd.DataFrame(
            {
                "measurement": [20, 21, 22, 23],
                "neighbor": [10, 10, 10, 11],
                "amount": [1.0, 2.0, 3.0, 4.0],
                "date": pd.Timestamp("2001-01-01"),
            }
        ),
        key="measurement",
        date="date",
    )
    baseline = extract(tables, edges, examples, 1)
    edges = [*edges, Relationship("neighbors", "measurements", "row", "neighbor")]
    enriched = extract(tables, edges, examples, 1)
    # "history" sorts before "measurements", so its native feature names
    # remain identical. Its sum/count must not triple for neighbor 10.
    assert_frame_equal(baseline, enriched[list(baseline)])
    value = next(c for c in enriched if c.endswith("amount_sum"))
    np.testing.assert_allclose(enriched[value].iloc[[0, 1, 3, 4]], [6, 4, 6, 4])


def test_bounded_context_sampling_is_independent_of_physical_row_order():
    tables, edges, examples = schema()
    history = tables["history"].source
    extra = history.iloc[[0]].copy()
    extra["event"], extra["value"] = 104, 7.0
    tables["history"] = replace(
        tables["history"], source=pd.concat([history, extra], ignore_index=True)
    )
    kwargs = dict(
        root_name="entities",
        entity_key="id",
        timestamp="time",
        path=("neighbors",),
        context_depth=1,
        max_neighbors=1,
        memory_limit="1GB",
    )
    first = extract_path(examples, tables, edges, **kwargs)
    shuffled = {
        name: replace(table, source=table.source.sample(frac=1, random_state=42))
        for name, table in tables.items()
    }
    second = extract_path(examples, shuffled, edges, **kwargs)
    assert_frame_equal(first, second)
    mean = next(c for c in first if c.endswith("value_avg"))
    total = next(c for c in first if c.endswith("value_sum"))
    np.testing.assert_allclose(first[mean], first[total])


def test_fitted_context_plan_is_independent_of_prediction_batch():
    import pickle

    tables, edges, examples = schema()
    history = tables["history"].source.copy()
    history["category"] = ["group_a", "future", "group_b", "group_c"]
    tables["history"] = replace(tables["history"], source=history)
    training = examples.copy()
    training["y"] = [0.2, 0.8, 0.3]
    training["ready"] = training.time + pd.Timedelta(days=1)
    family = LearnedMessageFamily(
        max_depth=1,
        neighbor_context_depth=1,
        neighbor_context_max_columns=4,
        iterations=8,
        thread_count=1,
        memory_limit="1GB",
    )
    family.fit_transform(
        training,
        examples,
        tables=tables,
        relationships=edges,
        root_name="entities",
        entity_key="id",
        timestamp="time",
        target="y",
        available_at="ready",
    )
    assert "plan" in family.context_plans_[("neighbors",)]
    family = pickle.loads(pickle.dumps(family))
    future = examples.iloc[:2].copy()
    future["time"] = pd.Timestamp("2005-01-01")
    full = family.transform(future, tables=tables, relationships=edges)
    alone = family.transform(future.iloc[:1], tables=tables, relationships=edges)
    for a, b in zip(full, alone):
        assert_frame_equal(a.iloc[:1], b)
