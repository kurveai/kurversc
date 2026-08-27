from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import kurversc
from kurversc.core import (
    _Workspace,
    _build_graph,
    _materialize_at_cutoff,
    _resource_block_for,
    _select_training_cutoffs,
)


def test_single_training_frame_is_always_latest_cutoff() -> None:
    values = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"])

    assert _select_training_cutoffs(values, 1) == (pd.Timestamp("2020-03-01"),)


def test_configuration_search_rejects_multiple_training_frames() -> None:
    with pytest.raises(ValueError, match="always uses the latest"):
        kurversc.fit(
            parent_node=pd.DataFrame(),
            label_node=pd.DataFrame(),
            search_training_frames=2,
        )


def test_resource_failure_blocks_deeper_and_wider_supersets_only() -> None:
    block = (("base", "temporal"), True, 2, "out of memory")

    assert (
        _resource_block_for(
            kurversc.GraphConfig(
                feature_families=("base", "temporal", "conditional"),
                depth=3,
                auto_annotate_features=True,
            ),
            [block],
        )
        == block
    )
    assert (
        _resource_block_for(
            kurversc.GraphConfig(
                feature_families=("base", "temporal"),
                depth=1,
                auto_annotate_features=True,
            ),
            [block],
        )
        is None
    )
    assert (
        _resource_block_for(
            kurversc.GraphConfig(
                feature_families=("base", "temporal"),
                depth=2,
                auto_annotate_features=False,
            ),
            [block],
        )
        is None
    )


def test_fit_runs_one_real_graphreduce_trial() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": range(40),
            "age": [20 + value % 17 for value in range(40)],
            "segment": ["a" if value % 3 else "b" for value in range(40)],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": range(120),
            "customer_id": [value // 3 for value in range(120)],
            "amount": [float(value % 11) for value in range(120)],
            "occurred_at": pd.date_range("2025-01-01", periods=120, freq="h"),
        }
    )
    labels = pd.DataFrame(
        {
            "customer_id": range(40),
            "churn": [value % 2 for value in range(40)],
            "split": ["train"] * 30 + ["validation"] * 10,
        }
    )

    result = kurversc.fit(
        parent_node=kurversc.Table(
            customers, name="customers", key="customer_id", prefix="customer"
        ),
        label_node=kurversc.Labels(
            labels, key="customer_id", target="churn", split="split"
        ),
        tables=[
            kurversc.Table(
                events,
                name="events",
                key="event_id",
                date="occurred_at",
                prefix="event",
            )
        ],
        relationships=[
            kurversc.Relationship(
                parent="customers",
                child="events",
                parent_key="customer_id",
                child_key="customer_id",
            )
        ],
        max_depth=1,
        feature_family_stages=[("base",)],
        auto_annotate_options=[True],
        model_params={"iterations": 8, "depth": 3},
    )

    assert result.metric == "roc_auc"
    assert result.best_config.feature_families == ("base",)
    assert result.best_config.depth == 1
    assert len(result.trials) == 1
    assert result.best_trial.train_rows == 30
    assert result.best_trial.validation_rows == 10
    assert result.best_trial.feature_count > 2
    assert list(result.results["metric"]) == ["roc_auc"]

    prediction_rows = labels.loc[
        labels["split"] == "validation", ["customer_id"]
    ]
    predictions = kurversc.predict(
        result,
        parent_node=kurversc.Table(
            customers, name="customers", key="customer_id", prefix="customer"
        ),
        prediction_node=kurversc.Labels(prediction_rows, key="customer_id"),
        tables=[
            kurversc.Table(
                events,
                name="events",
                key="event_id",
                date="occurred_at",
                prefix="event",
            )
        ],
        relationships=[
            kurversc.Relationship(
                parent="customers",
                child="events",
                parent_key="customer_id",
                child_key="customer_id",
            )
        ],
        use_validation_model=True,
    )
    assert len(predictions) == 10
    assert predictions["prediction"].between(0, 1).all()


def test_fit_requires_join_keys() -> None:
    frame = pd.DataFrame({"id": [1, 2], "label": [0, 1]})

    try:
        kurversc.fit(frame, frame, target="label")
    except ValueError as exc:
        assert "parent_key and label_key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("fit should reject ambiguous keys")


def test_exact_one_call_api_accepts_file_paths(tmp_path: Path) -> None:
    parent_path = tmp_path / "entities.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "entity_id": range(30),
            "signal": [float(value % 7) for value in range(30)],
        }
    ).to_csv(parent_path, index=False)
    pd.DataFrame(
        {
            "entity_id": range(30),
            "outcome": [value % 2 for value in range(30)],
            "split": ["train"] * 20 + ["validation"] * 10,
        }
    ).to_csv(labels_path, index=False)

    result = kurversc.fit(
        parent_node=parent_path,
        label_node=labels_path,
        parent_key="entity_id",
        label_key="entity_id",
        target="outcome",
        split_column="split",
        max_depth=1,
        feature_family_stages=[("base",)],
        auto_annotate_options=[True],
        model_params={"iterations": 5, "depth": 2},
    )

    assert result.task == "classification"
    assert result.best_trial.status == "completed"


def test_temporal_fit_filters_parent_before_feature_generation() -> None:
    parent = pd.DataFrame(
        {
            "entity_id": [1, 2, 3, 4],
            "created_at": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-02-01"]
            ),
            "signal": [1.0, 2.0, 3.0, 4.0],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "timestamp": pd.to_datetime(["2020-01-10"] * 4 + ["2020-02-10"] * 4),
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "split": ["train"] * 4 + ["validation"] * 4,
        }
    )

    result = kurversc.fit(
        parent_node=kurversc.Table(
            parent,
            name="entities",
            key="entity_id",
            date="created_at",
        ),
        label_node=kurversc.Labels(
            labels,
            key="entity_id",
            target="target",
            timestamp="timestamp",
            split="split",
        ),
        max_depth=1,
        feature_family_stages=[("base",)],
        auto_annotate_options=[False],
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    # Entity 4 does not exist at the training cutoff but does at validation.
    assert result.best_trial.train_rows == 3
    assert result.best_trial.validation_rows == 4


def test_temporal_fit_refuses_parent_without_date_key() -> None:
    parent = pd.DataFrame({"entity_id": [1, 2], "signal": [1.0, 2.0]})
    labels = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="parent_date is required"):
        kurversc.fit(
            parent_node=kurversc.Table(parent, key="entity_id"),
            label_node=kurversc.Labels(
                labels,
                key="entity_id",
                target="target",
                timestamp="timestamp",
            ),
        )


def test_temporal_fit_accepts_explicitly_timeless_parent() -> None:
    parent = pd.DataFrame(
        {"entity_id": range(20), "signal": [float(value) for value in range(20)]}
    )
    labels = pd.DataFrame(
        {
            "entity_id": list(range(20)) * 2,
            "timestamp": pd.to_datetime(["2020-01-01"] * 20 + ["2020-02-01"] * 20),
            "target": [value % 2 for value in range(20)] * 2,
            "split": ["train"] * 20 + ["validation"] * 20,
        }
    )

    result = kurversc.fit(
        parent_node=kurversc.Table(parent, key="entity_id", timeless=True),
        label_node=kurversc.Labels(
            labels,
            key="entity_id",
            target="target",
            timestamp="timestamp",
            split="split",
        ),
        max_depth=1,
        feature_family_stages=[("base",)],
        auto_annotate_options=[False],
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    assert result.best_trial.train_rows == 20
    assert result.best_trial.validation_rows == 20


def test_dated_parent_always_receives_graphreduce_filter_ops() -> None:
    connection = duckdb.connect(":memory:")
    workspace = _Workspace(connection, sample_rows=100, random_state=42)
    table = kurversc.Table(
        pd.DataFrame(
            {
                "Id": [1],
                "CreationDate": pd.to_datetime(["2020-01-01"]),
                "value": [2.0],
            }
        ),
        name="entities",
        key="Id",
        date="CreationDate",
        prefix="entity",
    )
    workspace.add("entities", table.source)
    try:
        graph = _build_graph(
            workspace,
            {"entities": table},
            (),
            "entities",
            kurversc.GraphConfig(),
            cut_date=datetime(2020, 1, 2),
            compute_period_days=3650,
            excluded_columns=set(),
            root_entity_keys=pd.DataFrame({"Id": [1]}),
        )
        predicates = [operation.opval for operation in graph.parent_node.do_filters_ops]
        node_configuration = {
            "feature_families": graph.parent_node.feature_families,
            "auto_annotate_features": graph.parent_node.auto_annotate_features,
            "ts_periods": graph.parent_node.ts_periods,
            "categorical_top_k": graph.parent_node.categorical_top_k,
            "auto_text_features": graph.parent_node.auto_text_features,
        }
        filtered_root_rows = connection.sql(
            f'SELECT COUNT(*) FROM "{graph.parent_node.fpath}"'
        ).fetchone()[0]
    finally:
        workspace.close()
        connection.close()

    assert predicates == [
        "entity_Id IS NOT NULL",
        "entity_CreationDate <= TIMESTAMP '2020-01-02 00:00:00'",
    ]
    assert graph.cut_date == datetime(2020, 1, 2, 0, 0, 0, 1)
    assert filtered_root_rows == 1
    assert node_configuration == {
        "feature_families": ("base",),
        "auto_annotate_features": True,
        "ts_periods": [7, 30, 90],
        "categorical_top_k": 5,
        "auto_text_features": False,
    }


def test_relation_cutoff_is_relbench_inclusive_without_future_rows() -> None:
    connection = duckdb.connect(":memory:")
    workspace = _Workspace(connection, sample_rows=100, random_state=42)
    tables = {
        "entities": kurversc.Table(
            pd.DataFrame({"entity_id": [1]}),
            name="entities",
            key="entity_id",
            timeless=True,
            prefix="entity",
        ),
        "events": kurversc.Table(
            pd.DataFrame(
                {
                    "event_id": [1, 2, 3],
                    "entity_id": [1, 1, 1],
                    "occurred_at": pd.to_datetime(
                        [
                            "2020-01-01 23:59:59",
                            "2020-01-02 00:00:00",
                            "2020-01-02 00:00:00.000001",
                        ],
                        format="mixed",
                    ),
                }
            ),
            name="events",
            key="event_id",
            date="occurred_at",
            prefix="event",
        ),
    }
    labels = pd.DataFrame(
        {
            "entity_id": [1],
            "timestamp": pd.to_datetime(["2020-01-02"]),
            "target": [1],
        }
    )
    label_spec = kurversc.Labels(
        labels,
        key="entity_id",
        target="target",
        timestamp="timestamp",
    )
    for name, table in tables.items():
        workspace.add(name, table.source)
    try:
        frame = _materialize_at_cutoff(
            workspace,
            tables,
            (
                kurversc.Relationship(
                    parent="entities",
                    child="events",
                    parent_key="entity_id",
                    child_key="entity_id",
                ),
            ),
            "entities",
            kurversc.GraphConfig(auto_annotate_features=False),
            labels,
            label_spec,
            cut_date=datetime(2020, 1, 2),
            compute_period_days=3650,
            verbose=False,
        )
    finally:
        workspace.close()
        connection.close()

    assert frame.loc[0, "event_id_count"] == 2


def test_fit_uses_graphreduce_native_label_generation(monkeypatch) -> None:
    events_logged = []

    class CapturingLogger:
        def info(self, event, **values):
            events_logged.append((event, values))

        def warning(self, event, **values):
            events_logged.append((event, values))

    monkeypatch.setattr("kurversc.core.logger", CapturingLogger())
    entity_ids = list(range(20))
    events = []
    event_id = 0
    for entity_id in entity_ids:
        events.append((event_id, entity_id, "2020-01-15", float(entity_id)))
        event_id += 1
        if entity_id % 2 == 0:
            events.append((event_id, entity_id, "2020-02-10", 10.0))
            event_id += 1
        if entity_id % 3 == 0:
            events.append((event_id, entity_id, "2020-04-10", 20.0))
            event_id += 1
    event_frame = pd.DataFrame(
        events, columns=["event_id", "entity_id", "occurred_at", "amount"]
    )
    event_frame["occurred_at"] = pd.to_datetime(event_frame["occurred_at"])

    result = kurversc.fit(
        parent_node=kurversc.Table(
            pd.DataFrame({"entity_id": entity_ids}),
            name="entities",
            key="entity_id",
            timeless=True,
        ),
        label_node=kurversc.GraphLabels(
            table="events",
            field="event_id",
            operation="bool",
            period_days=30,
            train_cutoffs=("2020-02-01",),
            validation_cutoffs=("2020-04-01",),
            target="will_event",
        ),
        tables=[
            kurversc.Table(
                event_frame,
                name="events",
                key="event_id",
                date="occurred_at",
            )
        ],
        relationships=[
            kurversc.Relationship(
                parent="entities",
                child="events",
                parent_key="entity_id",
                child_key="entity_id",
            )
        ],
        max_depth=1,
        feature_family_stages=[("base",)],
        auto_annotate_options=[False],
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    assert result.task == "classification"
    assert result.best_trial.train_rows == 20
    assert result.best_trial.validation_rows == 20
    event_names = [event for event, _ in events_logged]
    assert "trial_started" in event_names
    assert "trial_completed" in event_names
    assert "search_selected" in event_names


def test_full_refit_uses_multiple_frames_and_frozen_plan_for_test() -> None:
    entity_ids = list(range(20))
    events: list[tuple[int, int, str, float]] = []
    event_id = 0
    for entity_id in entity_ids:
        events.append((event_id, entity_id, "2020-01-15", float(entity_id)))
        event_id += 1
        for event_date, selected in (
            ("2020-02-10", entity_id % 2 == 0),
            ("2020-03-10", entity_id % 3 == 0),
            ("2020-04-10", entity_id % 2 == 1),
        ):
            if selected:
                events.append((event_id, entity_id, event_date, 10.0))
                event_id += 1
    event_frame = pd.DataFrame(
        events, columns=["event_id", "entity_id", "occurred_at", "amount"]
    )
    event_frame["occurred_at"] = pd.to_datetime(event_frame["occurred_at"])

    result = kurversc.fit(
        parent_node=kurversc.Table(
            pd.DataFrame({"entity_id": entity_ids}),
            name="entities",
            key="entity_id",
            timeless=True,
        ),
        label_node=kurversc.GraphLabels(
            table="events",
            field="event_id",
            operation="bool",
            period_days=20,
            train_cutoffs=("2020-02-01", "2020-03-01"),
            validation_cutoffs=("2020-04-01",),
            test_cutoffs=("2020-05-01",),
            target="will_event",
        ),
        tables=[
            kurversc.Table(
                event_frame,
                name="events",
                key="event_id",
                date="occurred_at",
            )
        ],
        relationships=[
            kurversc.Relationship(
                parent="entities",
                child="events",
                parent_key="entity_id",
                child_key="entity_id",
            )
        ],
        sample_rows=10,
        search_training_frames=1,
        full_training_frames=2,
        max_depth=1,
        feature_family_stages=[("base",)],
        auto_annotate_options=[False],
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    assert result.best_trial.train_rows == 10
    assert result.fitted_model is not None
    assert result.fitted_model.training_frames == 2
    assert result.fitted_model.train_rows == 40
    assert result.fitted_model.validation_rows == 20
    assert result.execution_plan is result.fitted_model.execution_plan
    assert len(result.execution_plan["records"]) > 0
    assert result.test_predictions is not None
    assert len(result.test_predictions) == 20
    assert list(result.test_predictions.columns) == [
        "entity_id",
        "__kurversc_cutoff__",
        "prediction",
    ]
