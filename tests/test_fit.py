from __future__ import annotations

import inspect
import multiprocessing
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import kurversc
from kurversc.core import (
    _Workspace,
    _build_graph,
    _connection_scope,
    _fork_process_map,
    _load_frame_source,
    _materialize_at_cutoff,
    _observed_feature_width_estimate,
    _ordered_materialization_labels,
    _rerank_objective,
    _resource_block_for,
    _select_label_training_cutoffs,
    _select_temporal_rerank_trials,
    _select_trials,
    _select_training_cutoffs,
    _sample_temporal_frame,
    _table_workspace,
    _temporal_frame_row_limit,
    _with_graph_feature_options,
)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"high_cardinality_policy": "top_k"},
        {"text_categorical_fallback": True},
        {"auto_text_features": True},
        {"auto_annotate_max_text_columns": 2},
    ],
)
def test_graph_feature_options_preserve_unrelated_explicit_policies(options) -> None:
    config = kurversc.GraphConfig(
        auto_text_features=True,
        auto_annotate_max_text_columns=2,
        high_cardinality_policy="top_k",
        categorical_low_cardinality_policy="all",
        text_categorical_fallback=True,
    )
    assert _with_graph_feature_options((config,), **options) == (config,)


def test_graph_feature_options_keep_the_implicit_native_recipe() -> None:
    configs = kurversc.incremental_configs(max_depth=2)
    expected = tuple(replace(config, high_cardinality_policy="top_k",
        text_categorical_fallback=True) for config in configs)
    assert _with_graph_feature_options(configs, high_cardinality_policy="top_k",
        text_categorical_fallback=True) == expected
    assert _with_graph_feature_options(configs) == configs


def test_explicit_representation_survives_fit_and_frozen_plan_replay() -> None:
    config = kurversc.GraphConfig(
        auto_annotate_features=False,
        auto_annotate_max_text_columns=2,
        high_cardinality_policy="top_k",
        categorical_low_cardinality_policy="all",
        text_categorical_fallback=True,
        include_target_history=False,
    )
    arguments = dict(
        parent_node=pd.DataFrame({"id": range(20), "value": range(20)}),
        parent_key="id",
        parent_timeless=True,
        label_node=kurversc.Labels(pd.DataFrame({"id": range(20),
            "target": [0, 1] * 10,
            "split": ["train"] * 12 + ["validation"] * 8}),
            key="id", target="target", split="split", target_history=False),
        high_cardinality_policy="top_k",
        text_categorical_fallback=True,
        rerank_top_k=0,
        search_full_data=True,
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )
    selected = kurversc.fit(**arguments, graph_configs=(config,))
    assert selected.recommended_config == config
    replay = kurversc.fit(**arguments, preselected_config=config,
        preselected_execution_plan=selected.execution_plan)
    assert replay.recommended_config == config
    assert replay.plan_fingerprint == selected.plan_fingerprint
    assert replay.full_validation_score == selected.full_validation_score


def test_text_preflight_keeps_a_candidates_independent_text_budget() -> None:
    config = kurversc.GraphConfig(auto_text_features=True,
        auto_annotate_max_text_columns=2, include_target_history=False)
    preflight = kurversc.estimate_features(
        parent_node=pd.DataFrame({"id": range(10), "value": range(10)}),
        parent_key="id", parent_timeless=True, graph_configs=(config,),
        auto_text_features=True)
    assert preflight.estimates[0].config == config


def test_rerank_stability_penalty_uses_standard_error_of_mean() -> None:
    episode_objective, episode_se = _rerank_objective(
        "roc_auc", 0.676471, 0.040216, 3, 0.25
    )
    sequence_objective, sequence_se = _rerank_objective(
        "roc_auc", 0.673117, 0.019848, 3, 0.25
    )

    assert episode_se == pytest.approx(0.040216 / 3**0.5)
    assert sequence_se == pytest.approx(0.019848 / 3**0.5)
    assert episode_objective > sequence_objective


def test_connection_scope_accepts_unbounded_duckdb_resources() -> None:
    with _connection_scope(
        None,
        memory_limit=None,
        max_temp_directory_size=None,
    ) as connection:
        assert connection.sql("SELECT 1").fetchone() == (1,)


def test_workspace_dataframe_sampling_is_representative_nested_and_deterministic() -> (
    None
):
    source = pd.DataFrame(
        {
            "row_id": range(100),
            # RelBench search labels are deliberately ordered this way.
            "split": ["train"] * 60 + ["validation"] * 40,
        }
    )

    def sampled(limit: int) -> pd.DataFrame:
        connection = duckdb.connect(":memory:")
        workspace = _Workspace(connection, sample_rows=limit, random_state=42)
        try:
            workspace.add("labels", source)
            return workspace.frame("labels")
        finally:
            workspace.close()
            connection.close()

    first_ten = sampled(10)
    repeated_ten = sampled(10)
    first_twenty = sampled(20)

    assert first_ten.equals(repeated_ten)
    assert set(first_ten["row_id"]) <= set(first_twenty["row_id"])
    assert set(first_ten["split"]) == {"train", "validation"}
    assert set(first_ten["row_id"]) != set(range(10))


def test_full_search_keeps_the_uncapped_task_connected_source() -> None:
    table = kurversc.Table(
        pd.DataFrame({"entity_id": range(100)}),
        name="entities",
        key="entity_id",
        timeless=True,
        search_source=pd.DataFrame({"entity_id": range(10, 30)}),
    )

    with _table_workspace(
        None,
        {"entities": table},
        sample_rows=5,
        random_state=42,
        search=True,
        search_full_data=True,
        duckdb_memory_limit="1GB",
        duckdb_max_temp_directory_size="1GB",
    ) as workspace:
        frame = workspace.frame("entities")

    assert frame["entity_id"].tolist() == list(range(10, 30))


def test_full_search_keeps_the_uncapped_task_connected_label_cohort(
    monkeypatch,
) -> None:
    entities = pd.DataFrame(
        {
            "entity_id": range(30),
            "signal": [float(value % 7) for value in range(30)],
        }
    )
    production_labels = pd.DataFrame(
        {
            "entity_id": range(30),
            "outcome": [value % 2 for value in range(30)],
            "split": ["train"] * 20 + ["validation"] * 10,
        }
    )
    task_labels = production_labels.iloc[::-1].reset_index(drop=True)
    loaded_search_sources: list[tuple[object, bool]] = []

    def recording_load(source, **kwargs):
        if kwargs["name"] == "search_labels":
            loaded_search_sources.append((source, kwargs["sample"]))
        return _load_frame_source(source, **kwargs)

    monkeypatch.setattr("kurversc.core._load_frame_source", recording_load)
    result = kurversc.fit(
        parent_node=entities,
        label_node=kurversc.Labels(
            production_labels,
            key="entity_id",
            target="outcome",
            split="split",
            search_source=task_labels,
            target_history=False,
        ),
        parent_key="entity_id",
        parent_timeless=True,
        graph_configs=(kurversc.GraphConfig(include_target_history=False),),
        search_full_data=True,
        rerank_top_k=0,
        search_only=True,
        model_params={"iterations": 5, "depth": 2},
    )

    assert result.trials[0].status == "completed"
    assert len(loaded_search_sources) == 1
    assert loaded_search_sources[0][0] is task_labels
    assert loaded_search_sources[0][1] is False


def test_fidelity_workspace_uses_dedicated_source_without_changing_search_source() -> (
    None
):
    table = kurversc.Table(
        pd.DataFrame({"entity_id": [0]}),
        name="entities",
        key="entity_id",
        timeless=True,
        search_source=pd.DataFrame({"entity_id": [1]}),
        fidelity_source=pd.DataFrame({"entity_id": [1, 2, 3]}),
    )

    with _table_workspace(
        None,
        {"entities": table},
        sample_rows=10,
        random_state=42,
        search=True,
        fidelity=False,
        search_full_data=True,
        duckdb_memory_limit="1GB",
        duckdb_max_temp_directory_size="1GB",
    ) as workspace:
        search_values = workspace.frame("entities")["entity_id"].tolist()
    with _table_workspace(
        None,
        {"entities": table},
        sample_rows=10,
        random_state=42,
        search=True,
        fidelity=True,
        search_full_data=True,
        duckdb_memory_limit="1GB",
        duckdb_max_temp_directory_size="1GB",
    ) as workspace:
        fidelity_values = workspace.frame("entities")["entity_id"].tolist()

    assert search_values == [1]
    assert fidelity_values == [1, 2, 3]


def test_fidelity_promotion_reads_additional_labels_but_replays_search_plan() -> None:
    entities = pd.DataFrame(
        {
            "entity_id": range(30),
            "signal": [float(value % 5) for value in range(30)],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": range(30),
            "timestamp": pd.to_datetime(
                ["2020-01-01"] * 10 + ["2020-02-01"] * 10 + ["2020-03-01"] * 10
            ),
            "outcome": [0, 1] * 15,
            "split": ["train"] * 20 + ["validation"] * 10,
        }
    )
    search_labels = pd.concat([labels.iloc[:10], labels.iloc[20:]])
    fidelity_labels = labels.copy()
    search_entities = pd.concat([entities.iloc[:10], entities.iloc[20:]])

    result = kurversc.fit(
        parent_node=kurversc.Table(
            entities,
            name="entities",
            key="entity_id",
            timeless=True,
            search_source=search_entities,
            fidelity_source=entities,
        ),
        label_node=kurversc.Labels(
            labels,
            key="entity_id",
            target="outcome",
            timestamp="timestamp",
            split="split",
            search_source=search_labels,
            fidelity_source=fidelity_labels,
            target_history=False,
        ),
        graph_configs=(
            kurversc.GraphConfig(
                feature_families=("base",),
                depth=1,
                auto_annotate_features=False,
                include_target_history=False,
            ),
        ),
        sample_rows=100,
        search_full_data=True,
        search_training_frames=1,
        fidelity_promotion_top_k=1,
        fidelity_promotion_training_frames=2,
        rerank_top_k=0,
        search_only=True,
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    assert result.trials[0].train_rows == 10
    assert len(result.confirmation_trials) == 1
    promoted = result.confirmation_trials[0]
    assert promoted.stage == "fidelity_promotion"
    assert promoted.train_rows == 20
    assert promoted.execution_plan == result.trials[0].execution_plan
    assert promoted.feature_columns == result.trials[0].feature_columns


def test_cost_calibrated_screening_can_skip_an_unhelpful_row_stage() -> None:
    entities = pd.DataFrame(
        {
            "entity_id": range(60),
            "signal": [float(value % 7) for value in range(60)],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": range(60),
            "outcome": [value % 2 for value in range(60)],
            "split": ["train"] * 40 + ["validation"] * 20,
        }
    )

    result = kurversc.fit(
        parent_node=entities,
        label_node=kurversc.Labels(
            labels,
            key="entity_id",
            target="outcome",
            split="split",
            target_history=False,
        ),
        parent_key="entity_id",
        parent_timeless=True,
        graph_configs=(
            kurversc.GraphConfig(
                auto_annotate_features=False,
                include_target_history=False,
            ),
            kurversc.GraphConfig(include_target_history=False),
        ),
        sample_rows=60,
        screening_rows=20,
        screening_cost_ratio_threshold=1_000_000.0,
        structural_promotion=True,
        confirmation_top_k=2,
        rerank_top_k=0,
        search_only=True,
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    assert len(result.trials) == 2
    assert all(trial.status == "completed" for trial in result.trials)
    assert all(trial.sample_rows == 60 for trial in result.trials)
    assert not result.confirmation_trials


def test_parallel_map_uses_isolated_overlapping_processes() -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork-based search workers require POSIX")

    def timed_worker(value: int) -> tuple[int, int, float, float]:
        started = time.monotonic()
        time.sleep(0.2)
        return value, os.getpid(), started, time.monotonic()

    results = _fork_process_map(timed_worker, range(4), max_workers=4)

    assert [result[0] for result in results] == list(range(4))
    assert len({result[1] for result in results}) == 4
    assert max(result[2] for result in results) < min(result[3] for result in results)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="production process search is exercised on its Linux deployment target",
)
def test_parallel_fit_returns_trials_and_frozen_plans() -> None:
    entities = pd.DataFrame(
        {
            "entity_id": range(30),
            "signal": [float(value % 7) for value in range(30)],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": range(30),
            "outcome": [value % 2 for value in range(30)],
            "split": ["train"] * 20 + ["validation"] * 10,
        }
    )
    result = kurversc.fit(
        parent_node=entities,
        label_node=labels,
        parent_key="entity_id",
        label_key="entity_id",
        target="outcome",
        split_column="split",
        graph_configs=(
            kurversc.GraphConfig(auto_annotate_features=True),
            kurversc.GraphConfig(auto_annotate_features=False),
        ),
        search_parallelism=2,
        search_full_data=True,
        rerank_top_k=0,
        model_params={"iterations": 5, "depth": 2},
    )

    assert len(result.trials) == 2
    assert all(trial.status == "completed" for trial in result.trials)
    assert all(trial.execution_plan for trial in result.trials)
    assert result.fitted_model is not None


def test_search_only_returns_trials_without_production_refit() -> None:
    entities = pd.DataFrame(
        {
            "entity_id": range(30),
            "signal": [float(value % 7) for value in range(30)],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": range(30),
            "outcome": [value % 2 for value in range(30)],
            "split": ["train"] * 20 + ["validation"] * 10,
        }
    )

    result = kurversc.fit(
        parent_node=entities,
        label_node=labels,
        parent_key="entity_id",
        label_key="entity_id",
        target="outcome",
        split_column="split",
        graph_configs=(kurversc.GraphConfig(),),
        search_full_data=True,
        rerank_top_k=0,
        search_only=True,
        model_params={"iterations": 5, "depth": 2},
    )

    assert len(result.trials) == 1
    assert result.trials[0].status == "completed"
    assert result.fitted_model is None


def test_fit_resumes_completed_search_trials_from_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    events_logged: list[tuple[str, dict[str, object]]] = []

    class CapturingLogger:
        def info(self, event, **values):
            events_logged.append((event, values))

        def warning(self, event, **values):
            events_logged.append((event, values))

    monkeypatch.setattr("kurversc.core.logger", CapturingLogger())
    entities = pd.DataFrame(
        {
            "entity_id": range(30),
            "signal": [float(value % 7) for value in range(30)],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": range(30),
            "outcome": [value % 2 for value in range(30)],
            "split": ["train"] * 20 + ["validation"] * 10,
        }
    )
    kwargs = {
        "parent_node": entities,
        "label_node": labels,
        "parent_key": "entity_id",
        "label_key": "entity_id",
        "target": "outcome",
        "split_column": "split",
        "graph_configs": (
            kurversc.GraphConfig(auto_annotate_features=True),
            kurversc.GraphConfig(auto_annotate_features=False),
        ),
        "search_full_data": True,
        "rerank_top_k": 0,
        "model_params": {"iterations": 5, "depth": 2},
        "search_checkpoint_path": tmp_path / "search",
        "search_checkpoint_key": "test/task/seed-0",
    }

    first = kurversc.fit(**kwargs)
    assert sum(event == "search_checkpoint_saved" for event, _ in events_logged) == 2

    events_logged.clear()
    second = kurversc.fit(**kwargs)

    assert sum(event == "trial_resumed" for event, _ in events_logged) == 2
    assert "trial_started" not in {event for event, _ in events_logged}
    assert [trial.validation_score for trial in second.trials] == pytest.approx(
        [trial.validation_score for trial in first.trials]
    )

    events_logged.clear()
    expanded = {
        **kwargs,
        "graph_configs": (
            *kwargs["graph_configs"],
            kurversc.GraphConfig(depth=2, auto_annotate_features=False),
        ),
    }
    third = kurversc.fit(**expanded)

    assert sum(event == "trial_resumed" for event, _ in events_logged) == 2
    assert sum(event == "trial_started" for event, _ in events_logged) == 1
    assert len(third.trials) == 3


def test_single_training_frame_is_always_latest_cutoff() -> None:
    values = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"])

    assert _select_training_cutoffs(values, 1) == (pd.Timestamp("2020-03-01"),)


def test_default_search_and_production_temporal_contracts_are_aligned() -> None:
    parameters = inspect.signature(kurversc.fit).parameters

    assert parameters["search_training_frames"].default is None
    assert parameters["full_training_frames"].default == 1
    assert parameters["fidelity_promotion_top_k"].default == 0
    assert parameters["fidelity_promotion_training_frames"].default is None
    assert parameters["categorical_refinement_top_k"].default == 0
    assert parameters["stage_candidate_budgets"].default is None
    assert parameters["rerank_complexity_classification_loss_cap"].default == 0.01
    assert parameters["rerank_complexity_regression_relative_loss_cap"].default == 0.01


def test_stage_budgets_cap_screening_and_seeded_refinement_evaluations() -> None:
    entities = pd.DataFrame(
        {
            "entity_id": range(40),
            "category": [f"category-{value % 8}" for value in range(40)],
        }
    )
    labels = pd.DataFrame(
        {
            "entity_id": range(40),
            "outcome": [value % 2 for value in range(40)],
            "split": ["train"] * 30 + ["validation"] * 10,
        }
    )
    result = kurversc.fit(
        parent_node=kurversc.Table(
            entities,
            name="entities",
            key="entity_id",
            timeless=True,
        ),
        label_node=kurversc.Labels(
            labels,
            key="entity_id",
            target="outcome",
            split="split",
            target_history=False,
        ),
        graph_configs=(
            kurversc.GraphConfig(
                auto_annotate_features=False,
                include_target_history=False,
            ),
            kurversc.GraphConfig(
                depth=2,
                auto_annotate_features=False,
                include_target_history=False,
            ),
        ),
        search_full_data=True,
        categorical_refinement_top_k=1,
        categorical_refinement_seeds=(0, 1, 2),
        stage_candidate_budgets={
            "screening": 1,
            "categorical_refinement": 2,
            "rerank": 0,
        },
        rerank_top_k=3,
        search_only=True,
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    assert len(result.trials) == 1
    refinement_trials = [
        trial
        for trial in result.confirmation_trials
        if trial.stage == "categorical_refinement"
    ]
    assert len(refinement_trials) == 2
    assert [trial.config.categorical_tie_break_seed for trial in refinement_trials] == [
        0,
        1,
    ]
    assert all(
        trial.config.categorical_tie_break == "hash" for trial in refinement_trials
    )


def test_materialization_labels_have_canonical_supervised_order() -> None:
    labels = pd.DataFrame(
        {
            "entity_id": [3, 1, 2],
            "timestamp": pd.to_datetime(["2020-01-01"] * 3),
            "target": [0, 1, 0],
        }
    )
    ordered, order_column = _ordered_materialization_labels(
        labels,
        kurversc.Labels(
            labels,
            key="entity_id",
            target="target",
            timestamp="timestamp",
        ),
    )

    assert ordered["entity_id"].tolist() == [1, 2, 3]
    assert ordered[order_column].tolist() == [0, 1, 2]


def test_materialization_labels_preserve_explicit_prediction_order() -> None:
    order_column = "__kurversc_prediction_order__"
    labels = pd.DataFrame(
        {
            "entity_id": [1, 2, 3],
            order_column: [2, 0, 1],
        }
    )
    ordered, _ = _ordered_materialization_labels(
        labels,
        kurversc.Labels(labels, key="entity_id"),
    )

    assert ordered[order_column].tolist() == [0, 1, 2]
    assert ordered["entity_id"].tolist() == [2, 3, 1]


def test_published_materialization_preserves_training_merge_order_only() -> None:
    connection = duckdb.connect(":memory:")
    workspace = _Workspace(connection, sample_rows=100, random_state=42)
    table = kurversc.Table(
        pd.DataFrame({"entity_id": [2, 1], "value": [20.0, 10.0]}),
        name="entities",
        key="entity_id",
        timeless=True,
        prefix="entity",
    )
    labels = pd.DataFrame(
        {
            "entity_id": [2, 1],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "target": [1, 0],
        }
    )
    label_spec = kurversc.Labels(
        labels,
        key="entity_id",
        target="target",
        timestamp="timestamp",
    )
    config = kurversc.GraphConfig(
        auto_annotate_features=False,
        include_target_history=False,
        published_cutoff_compatibility=True,
        legacy_training_row_order=True,
    )
    workspace.add("entities", table.source)
    try:
        training = _materialize_at_cutoff(
            workspace,
            {"entities": table},
            (),
            "entities",
            config,
            labels,
            label_spec,
            cut_date=datetime(2020, 1, 1),
            compute_period_days=3650,
            verbose=False,
            train=True,
        )
        prediction = _materialize_at_cutoff(
            workspace,
            {"entities": table},
            (),
            "entities",
            config,
            labels,
            label_spec,
            cut_date=datetime(2020, 1, 1),
            compute_period_days=3650,
            verbose=False,
            train=False,
        )
    finally:
        workspace.close()
        connection.close()

    assert training["entity_id"].tolist() == [2, 1]
    assert prediction["entity_id"].tolist() == [1, 2]


def test_published_cutoff_can_use_canonical_training_row_order() -> None:
    connection = duckdb.connect(":memory:")
    workspace = _Workspace(connection, sample_rows=100, random_state=42)
    table = kurversc.Table(
        pd.DataFrame({"entity_id": [2, 1], "value": [20.0, 10.0]}),
        name="entities",
        key="entity_id",
        timeless=True,
        prefix="entity",
    )
    labels = pd.DataFrame(
        {
            "entity_id": [2, 1],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "target": [1, 0],
        }
    )
    label_spec = kurversc.Labels(
        labels,
        key="entity_id",
        target="target",
        timestamp="timestamp",
    )
    config = kurversc.GraphConfig(
        auto_annotate_features=False,
        include_target_history=False,
        published_cutoff_compatibility=True,
    )
    workspace.add("entities", table.source)
    try:
        training = _materialize_at_cutoff(
            workspace,
            {"entities": table},
            (),
            "entities",
            config,
            labels,
            label_spec,
            cut_date=datetime(2020, 1, 1),
            compute_period_days=3650,
            verbose=False,
            train=True,
        )
    finally:
        workspace.close()
        connection.close()

    assert training["entity_id"].tolist() == [1, 2]


def test_root_reachable_batched_materialization_matches_full_base_frame() -> None:
    roots = kurversc.Table(
        pd.DataFrame({"entity_id": [1, 2, 3, 4], "value": [1., 2., 3., 4.]}),
        key="entity_id", timeless=True, prefix="root",
    )
    events = kurversc.Table(
        pd.DataFrame({
            "entity_id": [1, 1, 2, 3, 4, 4],
            "when": pd.to_datetime(["2019-01-01"] * 6),
            "amount": [1., 2., 3., 4., 500., 600.],
        }),
        date="when", prefix="event",
    )
    tables = {"roots": roots, "events": events}
    edges = (kurversc.Relationship("roots", "events", "entity_id", "entity_id"),)
    labels = pd.DataFrame({
        "entity_id": [3, 1, 2],
        "timestamp": pd.to_datetime(["2020-01-01"] * 3),
        "target": [0, 1, 0],
    })
    spec = kurversc.Labels(labels, key="entity_id", target="target", timestamp="timestamp")
    ordinary = kurversc.GraphConfig(
        auto_annotate_features=False, include_target_history=False,
        base_distribution_features=False,
    )
    bounded = kurversc.GraphConfig(
        auto_annotate_features=False, include_target_history=False,
        base_distribution_features=False,
        materialization_batch_rows=2, root_reachable_materialization=True,
    )
    with _table_workspace(None, tables, sample_rows=100, random_state=0, search=False) as workspace:
        plans = []
        full = _materialize_at_cutoff(
            workspace, tables, edges, "roots", ordinary, labels, spec,
            cut_date=datetime(2020, 1, 1), compute_period_days=3650,
            verbose=False, frozen_plan_sink=plans,
        )
        batched = _materialize_at_cutoff(
            workspace, tables, edges, "roots", bounded, labels, spec,
            cut_date=datetime(2020, 1, 1), compute_period_days=3650,
            verbose=False, execution_plan=plans[0],
        )
        initial_plans = []
        batched_initial = _materialize_at_cutoff(
            workspace, tables, edges, "roots", bounded, labels, spec,
            cut_date=datetime(2020, 1, 1), compute_period_days=3650,
            verbose=False, frozen_plan_sink=initial_plans,
        )
    pd.testing.assert_frame_equal(full, batched, check_dtype=False)
    assert len(initial_plans) == 1
    assert batched_initial["root_entity_id"].tolist() == full["root_entity_id"].tolist()
    assert batched_initial["target"].tolist() == full["target"].tolist()


def test_fit_root_reachable_switch_preserves_base_search_score() -> None:
    entities = pd.DataFrame({"entity_id": range(30), "value": range(30)})
    labels = pd.DataFrame({
        "entity_id": range(30),
        "target": [index % 2 for index in range(30)],
        "split": ["train"] * 20 + ["validation"] * 10,
    })
    options = {
        "parent_node": entities,
        "parent_key": "entity_id",
        "parent_timeless": True,
        "label_node": kurversc.Labels(
            labels, key="entity_id", target="target", split="split",
            target_history=False,
        ),
        "graph_configs": (
            kurversc.GraphConfig(include_target_history=False),
        ),
        "rerank_top_k": 0,
        "search_only": True,
        "model_params": {"iterations": 5, "depth": 2},
    }
    ordinary = kurversc.fit(**options)
    reachable = kurversc.fit(**options, root_reachable_materialization=True)

    assert ordinary.trials[0].status == reachable.trials[0].status == "completed"
    assert not ordinary.trials[0].config.root_reachable_materialization
    assert reachable.trials[0].config.root_reachable_materialization
    assert ordinary.trials[0].validation_score == pytest.approx(
        reachable.trials[0].validation_score
    )


def test_classification_training_frames_prefer_class_valid_cutoffs() -> None:
    labels = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-03-01",
                    "2020-04-01",
                    "2020-05-01",
                    "2020-05-01",
                ]
            ),
            "target": [0, 1, 1, 0, 1, 1, 0, 1],
        }
    )

    selected = _select_label_training_cutoffs(
        labels,
        timestamp="timestamp",
        target="target",
        task="classification",
        limit=3,
    )

    assert selected == tuple(pd.to_datetime(["2020-01-01", "2020-03-01", "2020-05-01"]))


def test_configuration_search_rejects_nonpositive_training_frames() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        kurversc.fit(
            parent_node=pd.DataFrame(),
            label_node=pd.DataFrame(),
            search_training_frames=0,
        )


def test_fidelity_training_frames_require_promotion_candidates() -> None:
    with pytest.raises(ValueError, match="requires fidelity_promotion_top_k"):
        kurversc.fit(
            parent_node=pd.DataFrame(),
            label_node=pd.DataFrame(),
            fidelity_promotion_training_frames=3,
        )


def test_temporal_row_budget_is_a_true_global_cap() -> None:
    frame = pd.DataFrame({"feature": range(25)})
    labels = pd.Series([value % 2 for value in range(25)])
    sampled_rows = 0
    for frame_index in range(3):
        sampled, sampled_labels = _sample_temporal_frame(
            frame,
            labels,
            total_limit=10,
            frame_count=3,
            frame_index=frame_index,
            task="classification",
            random_state=42,
        )
        assert len(sampled) == len(sampled_labels)
        sampled_rows += len(sampled)

    assert sampled_rows == 10


def test_temporal_row_budget_requires_every_cutoff_to_be_represented() -> None:
    with pytest.raises(ValueError, match="at least the number of temporal frames"):
        _temporal_frame_row_limit(
            2,
            frame_count=3,
            frame_index=0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rerank_top_k", -1, "non-negative"),
        ("rerank_cutoff_frames", 0, "positive"),
        ("rerank_stability_penalty", -0.1, "non-negative"),
        ("screening_rows", 0, "positive"),
        ("confirmation_top_k", -1, "non-negative"),
        ("fidelity_promotion_top_k", -1, "non-negative"),
        ("fidelity_promotion_training_frames", 0, "positive or None"),
        ("search_parallelism", 0, "positive"),
        ("staged_search_top_k", -1, "non-negative"),
        (
            "rerank_complexity_classification_loss_cap",
            -0.1,
            "non-negative",
        ),
        (
            "rerank_complexity_regression_relative_loss_cap",
            -0.1,
            "non-negative",
        ),
        ("estimator_train_rows", 1, "at least 2"),
        ("temporal_estimator_mode", "unknown", "pooled.*frame_ensemble"),
        (
            "categorical_low_cardinality_policy",
            "unknown",
            "top_k.*all",
        ),
        ("search_max_features", 0, "positive"),
        ("search_max_working_cells", 0, "positive"),
        ("duckdb_memory_limit", "", "non-empty string"),
        ("duckdb_max_temp_directory_size", "", "non-empty string"),
        ("planner_collection_candidates", 0, "positive or None"),
        ("planner_collection_training_frames", 0, "must be positive"),
        ("planner_collection_column_budgets", (), "must not be empty"),
        ("target_history_options", (), "must not be empty"),
        (
            "max_base_table_occurrence_options",
            (0, 1),
            "must contain positive integers",
        ),
    ],
)
def test_reranking_settings_are_validated(field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        kurversc.fit(
            parent_node=pd.DataFrame(),
            label_node=pd.DataFrame(),
            **{field: value},
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


def test_observed_width_does_not_erase_topology_estimate() -> None:
    shallow = kurversc.Trial(
        config=kurversc.GraphConfig(
            feature_families=("base",),
            depth=1,
            auto_annotate_features=False,
        ),
        metric="roc_auc",
        validation_score=0.7,
        objective_score=0.7,
        feature_count=100,
        train_rows=100,
        validation_rows=100,
        feature_seconds=1.0,
        model_seconds=0.1,
    )
    deep = kurversc.GraphConfig(
        feature_families=("base",),
        depth=2,
        auto_annotate_features=False,
    )

    assert _observed_feature_width_estimate(deep, [shallow], audit_estimate=500) == 500


def test_complexity_selection_requires_gain_beyond_validation_uncertainty() -> None:
    simple = kurversc.Trial(
        config=kurversc.GraphConfig(feature_families=("base",)),
        metric="roc_auc",
        validation_score=0.800,
        objective_score=0.800,
        validation_standard_error=0.010,
        feature_count=20,
        train_rows=1_000,
        validation_rows=500,
        feature_seconds=1.0,
        model_seconds=1.0,
    )
    rich = kurversc.Trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        metric="roc_auc",
        validation_score=0.805,
        objective_score=0.805,
        validation_standard_error=0.010,
        feature_count=200,
        train_rows=1_000,
        validation_rows=500,
        feature_seconds=2.0,
        model_seconds=1.0,
    )

    best, recommended = _select_trials(
        [simple, rich],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
    )
    assert best is rich
    assert recommended is simple

    _, fixed_threshold_recommendation = _select_trials(
        [simple, rich],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=0.0,
    )
    assert fixed_threshold_recommendation is rich


def _temporal_rerank_trial(
    *,
    config: kurversc.GraphConfig,
    scores: tuple[float, ...],
    standard_errors: tuple[float, ...],
    features: int,
    metric: str = "roc_auc",
) -> kurversc.Trial:
    mean_score = sum(scores) / len(scores)
    score_std = (
        sum((score - mean_score) ** 2 for score in scores) / len(scores)
    ) ** 0.5
    objective_score, score_standard_error = _rerank_objective(
        metric, mean_score, score_std, len(scores), 0.25
    )
    return kurversc.Trial(
        config=config,
        metric=metric,
        validation_score=mean_score,
        objective_score=objective_score,
        validation_standard_error=score_standard_error,
        feature_count=features,
        train_rows=1_000,
        validation_rows=500,
        feature_seconds=1.0,
        model_seconds=1.0,
        stage="rerank",
        cutoff_labels=("historical-1", "historical-2", "recent"),
        cutoff_scores=scores,
        cutoff_standard_errors=standard_errors,
    )


def test_temporal_rerank_keeps_nondrastic_walk_forward_winner() -> None:
    simple = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",)),
        scores=(0.798, 0.803, 0.808),
        standard_errors=(0.010, 0.010, 0.020),
        features=20,
    )
    rich = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        scores=(0.800, 0.805, 0.810),
        standard_errors=(0.010, 0.010, 0.020),
        features=30,
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [simple, rich],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )
    assert best is rich
    assert recent_safe_best is rich
    assert recommended is rich


def test_temporal_rerank_complexity_guard_requires_drastic_growth() -> None:
    simple = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",)),
        scores=(0.798, 0.803, 0.808),
        standard_errors=(0.010, 0.010, 0.020),
        features=20,
    )
    rich = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        scores=(0.800, 0.805, 0.810),
        standard_errors=(0.010, 0.010, 0.020),
        features=200,
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [simple, rich],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )
    assert best is rich
    assert recent_safe_best is rich
    assert recommended is simple


def test_temporal_rerank_recent_fold_only_vetoes_material_loss() -> None:
    walk_forward_winner = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",)),
        scores=(0.900, 0.900, 0.750),
        standard_errors=(0.010, 0.010, 0.010),
        features=20,
    )
    recent_winner = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        scores=(0.600, 0.600, 0.800),
        standard_errors=(0.010, 0.010, 0.010),
        features=200,
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [walk_forward_winner, recent_winner],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )
    assert best is walk_forward_winner
    assert recent_safe_best is recent_winner
    assert recommended is recent_winner


def test_temporal_rerank_small_recent_difference_preserves_walk_forward_signal() -> (
    None
):
    recent_winner = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",)),
        scores=(0.800, 0.800, 0.808),
        standard_errors=(0.010, 0.010, 0.010),
        features=191,
    )
    walk_forward_winner = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "sequence"), depth=1),
        scores=(0.830, 0.830, 0.805),
        standard_errors=(0.010, 0.010, 0.010),
        features=335,
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [recent_winner, walk_forward_winner],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )
    assert best is walk_forward_winner
    assert recent_safe_best is walk_forward_winner
    assert recommended is walk_forward_winner


def test_temporal_rerank_recent_fold_uses_joint_uncertainty() -> None:
    state = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "state"), depth=1),
        scores=(0.875, 0.875, 0.652735),
        standard_errors=(0.020, 0.020, 0.025920),
        features=408,
    )
    context = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "context"), depth=2),
        scores=(0.846688, 0.846688, 0.669315),
        standard_errors=(0.020, 0.020, 0.025351),
        features=529,
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [state, context],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )

    # Context leads the latest fold by more than the old fixed 0.01 veto, but
    # not by the candidates' joint uncertainty.  The stronger, smaller state
    # graph must therefore survive the recent gate and win the stability tie.
    assert best is state
    assert recent_safe_best is state
    assert recommended is state


def test_practical_recent_tie_cannot_spend_two_auc_points_on_old_fold_mean() -> None:
    old_fold_leader = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",)),
        scores=(.944, .712, .649),
        standard_errors=(.059, .114, .026),
        features=250,
    )
    recent_leader = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "sequence", "context"), depth=2),
        scores=(.917, .702, .677),
        standard_errors=(.077, .116, .025),
        features=783,
    )
    for complexity_selector in (False, True):
        best, safe, selected = _select_temporal_rerank_trials(
            [old_fold_leader, recent_leader], classification_negligible_gain=.002,
            regression_negligible_relative_gain=.005, complexity_uncertainty_multiplier=1.,
            complexity_selector=complexity_selector, practical_recent_ties=True)
        assert best is old_fold_leader  # preserved historical audit, not the decision
        assert safe is recent_leader
        assert selected is recent_leader


def test_practical_recent_tie_preserves_stability_tiebreak_inside_loss_cap() -> None:
    stable = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "state")),
        scores=(.85, .85, .795), standard_errors=(.02, .02, .025), features=400)
    recent = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "context"), depth=2),
        scores=(.8, .8, .8), standard_errors=(.02, .02, .025), features=500)
    _, _, selected = _select_temporal_rerank_trials(
        [stable, recent], classification_negligible_gain=.002,
        regression_negligible_relative_gain=.005, complexity_uncertainty_multiplier=1.,
        complexity_selector=True, practical_recent_ties=True)
    assert selected is stable


def test_practical_recent_regression_tie_caps_relative_latest_loss() -> None:
    stable = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",)),
        scores=(5., 5., 10.3), standard_errors=(.5, .5, 1.), features=50, metric="mae")
    recent = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "sequence"), depth=2),
        scores=(9., 9., 10.), standard_errors=(.5, .5, 1.), features=500, metric="mae")
    _, _, selected = _select_temporal_rerank_trials(
        [stable, recent], classification_negligible_gain=.002,
        regression_negligible_relative_gain=.005, complexity_uncertainty_multiplier=1.,
        complexity_selector=True, practical_recent_ties=True)
    assert selected is recent


def test_temporal_rerank_complexity_guard_caps_top3_practical_loss() -> None:
    temporal = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        scores=(0.893333, 0.877778, 0.689300),
        standard_errors=(0.100, 0.100, 0.029),
        features=1_139,
    )
    state = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "state"), depth=1),
        scores=(0.893333, 0.800000, 0.724042),
        standard_errors=(0.100, 0.100, 0.028),
        features=411,
    )
    base = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",), depth=1),
        scores=(0.666667, 0.877778, 0.694478),
        standard_errors=(0.150, 0.098, 0.029),
        features=162,
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [temporal, state, base],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )

    # Marginal three-fold errors made the old complexity guard tolerate the
    # base graph's 0.074 mean-AUC loss. The hard 0.01 practical cap prevents
    # feature count from overruling a material predictive advantage.
    assert best is temporal
    assert recent_safe_best is temporal
    assert recommended is temporal


def test_temporal_rerank_complexity_guard_uses_paired_fold_differences() -> None:
    rich = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        scores=(0.700, 0.800, 0.900),
        standard_errors=(0.050, 0.050, 0.050),
        features=200,
    )
    simple = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",), depth=1),
        scores=(0.692, 0.792, 0.892),
        standard_errors=(0.050, 0.050, 0.050),
        features=20,
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [rich, simple],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )

    # The individual fold scores are noisy, but the rich graph wins every
    # paired fold by the same 0.008 AUC. Marginal standard errors would call
    # this a tie; the paired difference has zero uncertainty and preserves the
    # repeatable predictive advantage.
    assert best is rich
    assert recent_safe_best is rich
    assert recommended is rich


def test_temporal_rerank_regression_complexity_loss_is_capped_at_one_percent() -> None:
    rich = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=2),
        scores=(2.0, 3.0, 4.0),
        standard_errors=(0.5, 0.5, 0.5),
        features=200,
        metric="mae",
    )
    simple = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",), depth=1),
        scores=(2.5, 2.6, 4.08),
        standard_errors=(0.5, 0.5, 0.5),
        features=20,
        metric="mae",
    )

    best, recent_safe_best, recommended = _select_temporal_rerank_trials(
        [rich, simple],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
    )

    assert best is rich
    assert recent_safe_best is rich
    assert recommended is rich


def test_temporal_rerank_nested_depth_safety_rescues_shallow_control() -> None:
    shallow = _temporal_rerank_trial(
        config=kurversc.GraphConfig(
            feature_families=("base",), depth=1, auto_annotate_features=True
        ),
        scores=(0.944444, 0.615385, 0.577134),
        standard_errors=(0.058902, 0.126217, 0.028004),
        features=689,
    )
    deep = _temporal_rerank_trial(
        config=kurversc.GraphConfig(
            feature_families=("base",), depth=2, auto_annotate_features=True
        ),
        scores=(0.944444, 0.576923, 0.599646),
        standard_errors=(0.058902, 0.129346, 0.027470),
        features=1_596,
    )
    recent_leader = _temporal_rerank_trial(
        config=kurversc.GraphConfig(
            feature_families=("base", "temporal"),
            depth=1,
            auto_annotate_features=True,
        ),
        scores=(1.0, 0.509615, 0.618830),
        standard_errors=(0.0, 0.132548, 0.026957),
        features=1_189,
    )
    selection_kwargs = {
        "classification_negligible_gain": 0.002,
        "regression_negligible_relative_gain": 0.005,
        "complexity_uncertainty_multiplier": 1.0,
        "complexity_selector": True,
    }
    _, recent_safe_best, unguarded = _select_temporal_rerank_trials(
        [shallow, deep, recent_leader], **selection_kwargs
    )
    assert recent_safe_best is deep
    assert unguarded is deep

    _, _, guarded = _select_temporal_rerank_trials(
        [shallow, deep, recent_leader],
        nested_depth_safety=True,
        **selection_kwargs,
    )
    assert guarded is shallow


def test_temporal_rerank_nested_depth_safety_preserves_recent_family_winner() -> None:
    state = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "state"), depth=1),
        scores=(0.613333, 0.800000, 0.587098),
        standard_errors=(0.153358, 0.119670, 0.030101),
        features=411,
    )
    temporal = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base", "temporal"), depth=1),
        scores=(0.400000, 0.788889, 0.659234),
        standard_errors=(0.144222, 0.122016, 0.029550),
        features=654,
    )
    base_history = _temporal_rerank_trial(
        config=kurversc.GraphConfig(
            feature_families=("base",), depth=1, include_target_history=True
        ),
        scores=(0.466667, 0.777778, 0.602166),
        standard_errors=(0.150725, 0.124213, 0.030058),
        features=191,
    )
    _, _, guarded = _select_temporal_rerank_trials(
        [state, temporal, base_history],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
        nested_depth_safety=True,
    )
    assert guarded is temporal


def test_temporal_rerank_nested_depth_safety_preserves_clear_depth_gain() -> None:
    shallow = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",), depth=1),
        scores=(0.70, 0.70, 0.70),
        standard_errors=(0.01, 0.01, 0.01),
        features=100,
    )
    deep = _temporal_rerank_trial(
        config=kurversc.GraphConfig(feature_families=("base",), depth=2),
        scores=(0.80, 0.80, 0.80),
        standard_errors=(0.01, 0.01, 0.01),
        features=400,
    )
    _, _, guarded = _select_temporal_rerank_trials(
        [shallow, deep],
        classification_negligible_gain=0.002,
        regression_negligible_relative_gain=0.005,
        complexity_uncertainty_multiplier=1.0,
        complexity_selector=True,
        nested_depth_safety=True,
    )
    assert guarded is deep


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
        feature_family_max_column_options=(4,),
        infer_ts_periods=True,
        model_params={"iterations": 8, "depth": 3},
    )

    assert result.metric == "roc_auc"
    assert result.best_config.feature_families == ("base",)
    assert result.best_config.depth == 1
    assert len(result.trials) == 1
    assert result.best_trial.train_rows == 30
    assert result.best_trial.validation_rows == 10
    assert result.best_trial.feature_count > 2
    assert result.best_trial.stage == "confirmation"
    assert len(result.confirmation_trials) == 1
    assert result.confirmation_trials[0].sample_rows == 100_000
    assert not result.feature_audit.empty
    assert set(result.feature_audit["table"]) == {"customers", "events"}
    assert "eligible_budget_tiers" in result.feature_audit
    assert list(result.results["metric"]) == ["roc_auc"]
    assert result.fitted_model is not None
    assert result.execution_plan["kurversc_ts_periods"]
    assert result.execution_plan["kurversc_ts_periods"]["event"] != (
        7,
        30,
        90,
        3650,
    )

    prediction_rows = labels.loc[labels["split"] == "validation", ["customer_id"]]
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
            "entity_id": [1, 2, 3, 4] * 3,
            "timestamp": pd.to_datetime(
                ["2020-01-05"] * 4 + ["2020-01-10"] * 4 + ["2020-02-10"] * 4
            ),
            "target": [0, 1, 0, 1] * 3,
            "split": ["train"] * 8 + ["validation"] * 4,
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
        rerank_top_k=0,
    )

    # Entity 4 does not exist at the training cutoff but does at validation.
    assert result.best_trial.train_rows == 3
    assert result.best_trial.validation_rows == 4
    assert result.fitted_model is not None
    assert len(result.fitted_model.target_history) == len(labels)
    assert result.fitted_model.target_history_key == "entity_id"
    assert any(
        column.startswith("targethist_")
        for column in result.fitted_model.feature_columns
    )

    # Prediction reattaches train-plus-validation history from the artifact;
    # callers do not need to construct or pass the reserved child table.
    predictions = kurversc.predict(
        result,
        parent_node=kurversc.Table(
            parent,
            name="entities",
            key="entity_id",
            date="created_at",
        ),
        prediction_node=kurversc.Labels(
            pd.DataFrame(
                {
                    "entity_id": [1],
                    "timestamp": pd.to_datetime(["2020-03-10"]),
                }
            ),
            key="entity_id",
            timestamp="timestamp",
        ),
    )
    assert len(predictions) == 1
    assert predictions["prediction"].between(0, 1).all()


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
    from graphreduce.predicates import EqualityPredicate

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
        auto_base_predicate_max=1,
        base_predicate_windows=(14, 30),
        base_predicates=(EqualityPredicate("status", "invited"),),
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
            "feature_family_max_features_per_column": (
                graph.parent_node.feature_family_max_features_per_column
            ),
            "feature_propagation_max_functions_per_column": (
                graph.parent_node.feature_propagation_max_functions_per_column
            ),
            "auto_annotate_features": graph.parent_node.auto_annotate_features,
            "ts_periods": graph.parent_node.ts_periods,
            "categorical_top_k": graph.parent_node.categorical_top_k,
            "auto_text_features": graph.parent_node.auto_text_features,
            "auto_base_predicate_max": graph.parent_node.auto_base_predicate_max,
            "base_predicate_windows": graph.parent_node.base_predicate_windows,
            "base_predicates": graph.parent_node.base_predicates,
        }
        filtered_root_rows = connection.sql(
            f'SELECT COUNT(*) FROM "{graph.parent_node.fpath}"'
        ).fetchone()[0]
        published_graph = _build_graph(
            workspace,
            {"entities": table},
            (),
            "entities",
            kurversc.GraphConfig(
                include_target_history=False,
                published_cutoff_compatibility=True,
            ),
            cut_date=datetime(2020, 1, 2),
            compute_period_days=3650,
            excluded_columns=set(),
            root_entity_keys=pd.DataFrame({"Id": [1]}),
        )
        published_cut_date = published_graph.cut_date
        published_boundary = published_graph.parent_node.cutoff_boundary
    finally:
        workspace.close()
        connection.close()

    assert predicates == [
        "entity_Id IS NOT NULL",
        "entity_CreationDate <= TIMESTAMP '2020-01-02 00:00:00'",
    ]
    assert graph.cut_date == datetime(2020, 1, 2)
    assert graph.parent_node.cutoff_boundary == "inclusive"
    assert published_cut_date == datetime(2020, 1, 2, 0, 0, 0, 1)
    assert published_boundary == "strict"
    assert graph.infer_ts_periods is True
    assert filtered_root_rows == 1
    assert node_configuration == {
        "feature_families": ("base",),
        "feature_family_max_features_per_column": 32,
        "feature_propagation_max_functions_per_column": 1,
        "auto_annotate_features": True,
        "ts_periods": [1, 3, 4, 7, 14, 30, 60, 90, 180, 365, 730],
        "categorical_top_k": 5,
        "auto_text_features": False,
        "auto_base_predicate_max": 1,
        "base_predicate_windows": [14, 30],
        "base_predicates": (EqualityPredicate("status", "invited"),),
    }


def test_entity_filter_accepts_pandas_string_extension_keys() -> None:
    connection = duckdb.connect(":memory:")
    connection.sql("CREATE TABLE entities AS SELECT 'seller-1' AS entity_id")
    workspace = _Workspace(connection, sample_rows=100, random_state=42)
    workspace.add("entities", "entities")
    keys = pd.DataFrame({"entity_id": pd.Series(["seller-1"], dtype=pd.StringDtype())})

    try:
        view = workspace.entity_filtered_view(
            "entities", key_columns=("entity_id",), entity_keys=keys
        )
        rows = connection.sql(f'SELECT entity_id FROM "{view}"').fetchall()
    finally:
        workspace.close()
        connection.close()

    assert rows == [("seller-1",)]


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


def test_materialization_emits_context_features_before_child_reduction() -> None:
    connection = duckdb.connect(":memory:")
    workspace = _Workspace(connection, sample_rows=100, random_state=42)
    tables = {
        "drivers": kurversc.Table(
            pd.DataFrame({"driver_id": [1, 2]}),
            name="drivers",
            key="driver_id",
            timeless=True,
            prefix="driver",
        ),
        "results": kurversc.Table(
            pd.DataFrame(
                {
                    "result_id": [1, 2, 3],
                    "driver_id": [1, 2, 1],
                    "race_id": [10, 10, 11],
                    "occurred_at": pd.to_datetime(
                        ["2020-01-01", "2020-01-01", "2020-01-02"]
                    ),
                    "position": [1.0, 3.0, 4.0],
                }
            ),
            name="results",
            key="result_id",
            date="occurred_at",
            prefix="result",
            context_keys=("race_id",),
        ),
    }
    labels = pd.DataFrame(
        {
            "driver_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-03", "2020-01-03"]),
            "target": [1, 0],
        }
    )
    label_spec = kurversc.Labels(
        labels,
        key="driver_id",
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
                    parent="drivers",
                    child="results",
                    parent_key="driver_id",
                    child_key="driver_id",
                ),
            ),
            "drivers",
            kurversc.GraphConfig(
                feature_families=("base", "context"),
                auto_annotate_features=False,
            ),
            labels,
            label_spec,
            cut_date=datetime(2020, 1, 3),
            compute_period_days=3650,
            verbose=False,
        )
    finally:
        workspace.close()
        connection.close()

    context_columns = [column for column in frame if "__gr_context" in column]
    assert any("context_size" in column for column in context_columns)
    assert any("_delta" in column for column in context_columns)
    assert frame[context_columns].notna().any().all()


def test_materialization_emits_cutoff_safe_ordered_state_features() -> None:
    connection = duckdb.connect(":memory:")
    workspace = _Workspace(connection, sample_rows=100, random_state=42)
    tables = {
        "drivers": kurversc.Table(
            pd.DataFrame({"driver_id": [1, 2]}),
            name="drivers",
            key="driver_id",
            timeless=True,
            prefix="driver",
        ),
        "qualifying": kurversc.Table(
            pd.DataFrame(
                {
                    "qualifying_id": [1, 2, 3, 4, 5],
                    "driver_id": [1, 1, 1, 1, 2],
                    "constructor_id": [10, 10, 20, 20, 30],
                    "race_date": pd.to_datetime(
                        [
                            "2019-12-01",
                            "2020-01-01",
                            "2020-02-01",
                            "2020-04-01",  # after the cutoff: must be absent
                            "2020-02-01",
                        ]
                    ),
                    "position": [20.0, 8.0, 2.0, 1.0, 5.0],
                }
            ),
            name="qualifying",
            key="qualifying_id",
            date="race_date",
            prefix="qualifying",
            context_keys=("constructor_id",),
        ),
    }
    labels = pd.DataFrame(
        {
            "driver_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-03-01", "2020-03-01"]),
            "target": [1, 0],
        }
    )
    label_spec = kurversc.Labels(
        labels,
        key="driver_id",
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
                    parent="drivers",
                    child="qualifying",
                    parent_key="driver_id",
                    child_key="driver_id",
                ),
            ),
            "drivers",
            kurversc.GraphConfig(
                feature_families=("base", "state"),
                auto_annotate_features=False,
            ),
            labels,
            label_spec,
            cut_date=datetime(2020, 3, 1),
            compute_period_days=3650,
            verbose=False,
        )
    finally:
        workspace.close()
        connection.close()

    driver_1 = frame.loc[frame["driver_id"] == 1].iloc[0]
    assert driver_1["qualifying_position_state_latest"] == pytest.approx(2.0)
    assert driver_1["qualifying_position_state_lag2"] == pytest.approx(8.0)
    assert driver_1["qualifying_position_state_last3_avg"] == pytest.approx(10.0)
    assert driver_1["qualifying_constructor_id_state_latest"] == "20"


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


def test_search_buffers_multiple_frames_and_refit_replays_frozen_plan() -> None:
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
            ("2020-05-10", entity_id % 3 == 1),
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
            validation_cutoffs=("2020-04-01", "2020-05-01"),
            test_cutoffs=("2020-06-01",),
            target="will_event",
        ),
        tables=[
            kurversc.Table(
                event_frame,
                name="events",
                key="event_id",
                date="occurred_at",
                sample_in_search=False,
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
        # This test exercises temporal frame replay, so only sample the root;
        # retain the tiny event table to preserve both validation classes.
        sample_rows=10,
        search_training_frames=1,
        fidelity_promotion_top_k=1,
        fidelity_promotion_training_frames=2,
        full_training_frames=2,
        full_validation_frames=1,
        rerank_top_k=0,
        max_depth=1,
        feature_family_stages=[("base",)],
        auto_annotate_options=[False],
        model_params={"iterations": 5, "depth": 2},
        continue_on_error=False,
    )

    assert result.best_trial.train_rows == 20
    assert len(result.confirmation_trials) == 1
    assert result.confirmation_trials[0].stage == "fidelity_promotion"
    assert result.confirmation_trials[0].train_rows == 20
    assert (
        result.confirmation_trials[0].execution_plan == result.trials[0].execution_plan
    )
    assert (
        result.confirmation_trials[0].feature_columns
        == result.trials[0].feature_columns
    )
    assert result.fitted_model is not None
    assert result.fitted_model.training_frames == 2
    assert result.fitted_model.train_rows == 40
    assert result.fitted_model.validation_rows == 20
    assert not hasattr(result.fitted_model.estimator, "models")
    assert result.execution_plan is result.fitted_model.execution_plan
    assert len(result.execution_plan["records"]) > 0
    assert result.test_predictions is not None
    assert len(result.test_predictions) == 20
    assert list(result.test_predictions.columns) == [
        "entity_id",
        "__kurversc_cutoff__",
        "prediction",
    ]


@pytest.mark.parametrize(
    (
        "rerank_cutoff_frames",
        "temporal_estimator_mode",
        "full_training_frames",
        "scores",
        "expected_search_fit_rows",
        "expected_first_incremental_rows",
        "expected_total_incremental_rows",
        "expected_rerank_score",
        "expected_folds",
    ),
    [
        (
            3,
            "pooled",
            3,
            (0.800, 0.801, 0.600, 0.610, 0.620, 0.700, 0.900),
            (24, 24, 24, 48, 72),
            (72, 96),
            (72, 96, 72, 96),
            0.61,
            3,
        ),
        (
            1,
            "pooled",
            3,
            (0.800, 0.801, 0.600, 0.700, 0.900),
            (24, 24, 72),
            (72, 96),
            (72, 96, 72, 96),
            0.60,
            1,
        ),
        (
            3,
            "frame_ensemble",
            1,
            (0.800, 0.801, 0.600, 0.610, 0.620, 0.700, 0.900),
            (24, 24),
            (24,) * 5,
            (24,) * 7,
            0.61,
            3,
        ),
    ],
)
def test_full_frame_search_walk_forward_reranks_streaming_candidates(
    monkeypatch,
    rerank_cutoff_frames,
    temporal_estimator_mode,
    full_training_frames,
    scores,
    expected_search_fit_rows,
    expected_first_incremental_rows,
    expected_total_incremental_rows,
    expected_rerank_score,
    expected_folds,
) -> None:
    entity_ids = list(range(24))
    events: list[tuple[int, int, str, float]] = []
    event_id = 0
    for entity_id in entity_ids:
        events.append((event_id, entity_id, "2020-01-15", float(entity_id)))
        event_id += 1
        for event_date, selected in (
            ("2020-02-10", entity_id % 2 == 0),
            ("2020-03-10", entity_id % 3 == 0),
            ("2020-04-10", entity_id % 2 == 1),
            ("2020-05-10", entity_id % 4 == 0),
        ):
            if selected:
                events.append((event_id, entity_id, event_date, 10.0))
                event_id += 1
    event_frame = pd.DataFrame(
        events, columns=["event_id", "entity_id", "occurred_at", "amount"]
    )
    event_frame["occurred_at"] = pd.to_datetime(event_frame["occurred_at"])

    search_fit_rows: list[int] = []
    incremental_fit_rows: list[int] = []
    incremental_init_models: list[object | None] = []
    scores = iter(scores)

    def fake_search_fit(train_x, *args, **kwargs):
        del args, kwargs
        search_fit_rows.append(len(train_x))
        return object(), 0.01, (False, True)

    def fake_incremental_fit(train_x, *args, init_model=None, **kwargs):
        del args, kwargs
        incremental_fit_rows.append(len(train_x))
        incremental_init_models.append(init_model)
        return object(), 0.01, (False, True)

    monkeypatch.setattr("kurversc.core._fit_final_estimator", fake_search_fit)
    monkeypatch.setattr(
        "kurversc.core.fit_incremental_catboost_frame", fake_incremental_fit
    )
    monkeypatch.setattr(
        "kurversc.core._predict_values",
        lambda _model, features, *, task: pd.Series(
            0.5, index=features.index
        ).to_numpy(),
    )
    monkeypatch.setattr(
        "kurversc.core._score_prediction_chunks",
        lambda *args, **kwargs: ("roc_auc", next(scores)),
    )
    base = kurversc.GraphConfig(
        feature_families=("base",),
        depth=1,
        auto_annotate_features=False,
    )
    rich = kurversc.GraphConfig(
        feature_families=("base", "temporal"),
        depth=1,
        auto_annotate_features=False,
    )

    fit_kwargs = {
        "parent_node": kurversc.Table(
            pd.DataFrame({"entity_id": entity_ids}),
            name="entities",
            key="entity_id",
            timeless=True,
        ),
        "label_node": kurversc.GraphLabels(
            table="events",
            field="event_id",
            operation="bool",
            period_days=20,
            train_cutoffs=("2020-02-01", "2020-03-01", "2020-04-01"),
            validation_cutoffs=("2020-05-01",),
            target="will_event",
        ),
        "tables": [
            kurversc.Table(
                event_frame,
                name="events",
                key="event_id",
                date="occurred_at",
            )
        ],
        "relationships": [
            kurversc.Relationship(
                parent="entities",
                child="events",
                parent_key="entity_id",
                child_key="entity_id",
            )
        ],
        "sample_rows": 100,
        "search_training_frames": 1,
        "full_training_frames": full_training_frames,
        "estimator_train_rows": 100,
        "continue_on_error": False,
        "search_full_data": True,
        "rerank_top_k": 1,
        "rerank_cutoff_frames": rerank_cutoff_frames,
        "temporal_estimator_mode": temporal_estimator_mode,
    }
    result = kurversc.fit(**fit_kwargs, graph_configs=(base, rich))

    assert search_fit_rows == list(expected_search_fit_rows)
    assert incremental_fit_rows == list(expected_first_incremental_rows)
    assert all(model is None for model in incremental_init_models)
    assert result.trials[0].config == base
    assert result.best_config == base
    assert result.recommended_config == base
    assert [trial.config for trial in result.rerank_trials] == [base]
    assert result.rerank_trials[0].validation_score == pytest.approx(
        expected_rerank_score
    )
    assert all(
        f"folds={expected_folds}" in trial.note for trial in result.rerank_trials
    )
    assert result.fitted_model is not None
    assert result.fitted_model.config == base
    assert result.fitted_model.training_frames == full_training_frames
    assert result.full_validation_score == 0.70
    assert result.fitted_model.validation_estimator is not result.fitted_model.estimator
    assert (
        result.fitted_model.feature_columns == result.recommended_trial.feature_columns
    )

    preselected = kurversc.fit(
        **fit_kwargs,
        preselected_config=result.recommended_config,
        preselected_execution_plan=result.execution_plan,
    )

    assert search_fit_rows == list(expected_search_fit_rows)
    assert incremental_fit_rows == list(expected_total_incremental_rows)
    assert len(preselected.trials) == 1
    assert preselected.rerank_trials == ()
    assert preselected.plan_fingerprint == result.plan_fingerprint
    assert preselected.full_validation_score == 0.90


def test_frozen_plan_can_be_walk_forward_reranked_without_graph_search() -> None:
    entities = kurversc.Table(
        pd.DataFrame({"entity_id": range(24)}),
        name="entities",
        key="entity_id",
        timeless=True,
    )
    event_rows = [
        (entity_id, entity_id, "2020-01-15", float(entity_id % 7))
        for entity_id in range(24)
    ]
    for month, date in enumerate(
        ("2020-02-15", "2020-03-15", "2020-04-15", "2020-05-15"),
        start=1,
    ):
        event_rows.extend(
            (24 * month + entity_id, entity_id, date, float(entity_id % 7))
            for entity_id in range(24)
            if entity_id % 2 == month % 2
        )
    events = kurversc.Table(
        pd.DataFrame(
            event_rows,
            columns=["event_id", "entity_id", "occurred_at", "amount"],
        ).assign(occurred_at=lambda frame: pd.to_datetime(frame["occurred_at"])),
        name="events",
        key="event_id",
        date="occurred_at",
    )
    arguments = {
        "parent_node": entities,
        "label_node": kurversc.GraphLabels(
            table="events",
            field="event_id",
            operation="count",
            period_days=20,
            train_cutoffs=("2020-02-01", "2020-03-01", "2020-04-01"),
            validation_cutoffs=("2020-05-01",),
            target="event_count",
        ),
        "tables": [events],
        "relationships": [
            kurversc.Relationship(
                parent="entities",
                child="events",
                parent_key="entity_id",
                child_key="entity_id",
            )
        ],
        "sample_rows": 100,
        "search_training_frames": 1,
        "full_training_frames": 2,
        "search_full_data": True,
        "search_only": True,
        "model_params": {"iterations": 5, "depth": 2},
        "continue_on_error": False,
    }
    config = kurversc.GraphConfig(
        feature_families=("base",), depth=1, auto_annotate_features=False
    )
    original = kurversc.fit(**arguments, graph_configs=(config,), rerank_top_k=0)
    assert original.recommended_trial.execution_plan is not None

    temporal = kurversc.fit(
        **arguments,
        preselected_config=config,
        preselected_execution_plan=original.recommended_trial.execution_plan,
        rerank_preselected_config=True,
        rerank_top_k=1,
        rerank_cutoff_frames=3,
        temporal_estimator_mode="frame_ensemble",
    )

    assert len(temporal.rerank_trials) == 1
    assert temporal.rerank_trials[0].config == config
    assert len(temporal.rerank_trials[0].cutoff_scores) == 3
    assert temporal.fitted_model is None
