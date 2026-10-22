from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import kurversc
from examples.relbench_planner_collection import (
    _memory_limit_bytes,
    profile_relbench_database,
    profile_relbench_task,
)
from kurversc.planner_training import (
    KurvePlannerRanker,
    PlannerTrainingConfig,
    candidate_frame,
    evaluate_ranker,
    main as planner_main,
    summarize_episodes,
)


def _profile(database_id: str, *, events: float = 4.0) -> kurversc.DatabaseProfile:
    return kurversc.DatabaseProfile(
        database_id=database_id,
        as_of="2025-01-01T00:00:00+00:00",
        generated_at="2025-01-01T00:01:00+00:00",
        extractor="kurve",
        tables=(
            kurversc.TableProfile(
                table_id="users",
                row_count=100,
                primary_key=("user_id",),
                columns=(
                    kurversc.ColumnProfile(
                        name="user_id",
                        physical_type="integer",
                        role="primary_key",
                        distinct_count=100,
                        uniqueness_fraction=1.0,
                    ),
                ),
            ),
            kurversc.TableProfile(
                table_id="events",
                row_count=400,
                primary_key=("event_id",),
                date_key="event_at",
                time_span_seconds=31_536_000,
            ),
        ),
        relationships=(
            kurversc.RelationshipProfile(
                relationship_id="users-events",
                parent_table="users",
                child_table="events",
                parent_key=("user_id",),
                child_key=("user_id",),
                cardinality="one_to_many",
                avg_fanout=events,
                event_frequency=kurversc.EventFrequencyProfile(
                    num_dimensions_with_events=90,
                    avg_num_events=events,
                    median_num_events=events - 1,
                    avg_time_between_updates_seconds=86_400,
                ),
            ),
        ),
    )


def _task(
    task_id: str = "user-churn",
    *,
    task_type: str = "classification",
    metric: str = "roc_auc",
) -> kurversc.TaskProfile:
    return kurversc.TaskProfile(
        task_id=task_id,
        task_type=task_type,
        metric=metric,
        root_table="users",
        target_id="churn",
        support_rows=100,
        horizon_seconds=2_592_000,
        class_count=2,
        positive_fraction=0.2,
        probes={"recent_event_count_auc": 0.62},
    )


def _runtime() -> kurversc.RuntimeProfile:
    return kurversc.RuntimeProfile(
        downstream_model="catboost",
        cpu_count=4,
        memory_bytes=8_000_000_000,
        max_features=1_536,
    )


def test_explicit_relbench_profiler_is_cutoff_safe_and_value_free() -> None:
    from types import SimpleNamespace

    users = SimpleNamespace(
        df=pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "created_at": pd.to_datetime(["2020-01-01"] * 3),
                "segment": ["a", "b", "a"],
            }
        ),
        pkey_col="user_id",
        time_col="created_at",
        fkey_col_to_pkey_table={},
    )
    events = SimpleNamespace(
        df=pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4],
                "user_id": [1, 1, 2, 3],
                "event_at": pd.to_datetime(
                    ["2020-01-02", "2020-01-03", "2020-01-02", "2021-01-01"]
                ),
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        pkey_col="event_id",
        time_col="event_at",
        fkey_col_to_pkey_table={"user_id": "users"},
    )
    database = SimpleNamespace(table_dict={"users": users, "events": events})

    profile = profile_relbench_database(
        database,
        database_id="toy",
        cutoff="2020-12-31",
        profile_rows=100,
    )

    edge = profile.relationships[0]
    assert [table.row_count for table in profile.tables] == [3, 3]
    assert edge.cardinality == "one_to_many"
    assert edge.avg_fanout == 1.5
    assert edge.event_frequency is not None
    assert edge.event_frequency.avg_num_events == 1.5
    segment = next(
        column for column in profile.tables[0].columns if column.name == "segment"
    )
    assert segment.distinct_count == 2
    assert not hasattr(segment, "values")


def test_explicit_relbench_task_profile_uses_training_support_only() -> None:
    from types import SimpleNamespace

    labels = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-02-01", "2021-01-01"]),
            "churn": [False, True, True],
        }
    )
    task = SimpleNamespace(
        task_type="binary_classification",
        entity_table="users",
        entity_col="user_id",
        target_col="churn",
        time_col="timestamp",
        timedelta=pd.Timedelta(days=30),
        get_table=lambda split: SimpleNamespace(df=labels),
    )

    profile = profile_relbench_task(
        task,
        task_id="user-churn",
        cutoff="2020-12-31",
    )

    assert profile.support_rows == 2
    assert profile.positive_fraction == 0.5
    assert profile.horizon_seconds == 30 * 86_400


def test_teacher_runtime_memory_parser_accepts_duckdb_units() -> None:
    assert _memory_limit_bytes("48GB") == 48_000_000_000
    assert _memory_limit_bytes("2 GiB") == 2_000_000_000


def _episode(
    database_id: str,
    *,
    temporal_wins: bool = True,
    task_type: str = "classification",
) -> kurversc.PlannerEpisode:
    metric = "roc_auc" if task_type == "classification" else "mae"
    base_validation = 0.60 if temporal_wins else 0.75
    temporal_validation = 0.78 if temporal_wins else 0.61
    if task_type == "regression":
        base_validation = 0.60 if temporal_wins else 0.40
        temporal_validation = 0.40 if temporal_wins else 0.60
    base = kurversc.CandidateObservation(
        candidate_id="base",
        plan={
            "feature_families": ["base"],
            "depth": 1,
            "auto_annotate_features": True,
        },
        status="completed",
        supervision="observed",
        metric=metric,
        validation_score=base_validation,
        objective_score=(
            base_validation if task_type == "classification" else -base_validation
        ),
        estimated_feature_count=10,
    )
    temporal = kurversc.CandidateObservation(
        candidate_id="temporal",
        plan={
            "feature_families": ["base", "temporal"],
            "depth": 1,
            "auto_annotate_features": True,
        },
        status="completed",
        supervision="observed",
        metric=metric,
        validation_score=temporal_validation,
        objective_score=(
            temporal_validation
            if task_type == "classification"
            else -temporal_validation
        ),
        estimated_feature_count=30,
    )
    failed = kurversc.CandidateObservation(
        candidate_id="deep",
        plan={
            "feature_families": ["base", "temporal", "sequence"],
            "depth": 3,
            "auto_annotate_features": True,
        },
        status="failed",
        supervision="observed",
        metric=metric,
        validation_score=None,
        objective_score=None,
        estimated_feature_count=30_000,
        error_type="OutOfMemoryException",
    )
    return kurversc.PlannerEpisode(
        episode_id=f"{database_id}/user-churn/2025-01-01/screening",
        database_profile=_profile(database_id),
        task_profile=_task(task_type=task_type, metric=metric),
        runtime_profile=_runtime(),
        observations=(base, temporal, failed),
    )


def test_database_profile_round_trips_without_raw_values() -> None:
    profile = _profile("database-a")

    restored = kurversc.DatabaseProfile.from_dict(profile.to_dict())
    features = restored.aggregate_features()

    assert restored == profile
    assert features["schema_table_count"] == 2
    assert features["schema_event_frequency_edge_count"] == 1
    assert features["schema_avg_num_events_mean"] == 4.0


def test_database_profile_requires_a_valid_cutoff() -> None:
    with pytest.raises(ValueError, match="ISO-8601"):
        kurversc.DatabaseProfile(
            database_id="database-a",
            as_of="not-a-date",
            tables=(),
            relationships=(),
        )


def test_existing_kurve_event_payload_maps_without_sql_or_identifiers() -> None:
    frequency = kurversc.EventFrequencyProfile.from_kurve_payload(
        {
            "edge_id": 42,
            "child_node_identifier": "events",
            "sql": "SELECT private_values",
            "avg_num_events": 8.5,
            "median_num_events": 4.0,
            "avg_time_between_updates_seconds": 3_600,
        }
    )

    assert frequency.avg_num_events == 8.5
    assert frequency.avg_time_between_updates_seconds == 3_600
    assert "sql" not in frequency.__dict__


def test_fit_result_becomes_stage_aware_teacher_episodes() -> None:
    base = kurversc.GraphConfig(feature_families=("base",))
    temporal = kurversc.GraphConfig(feature_families=("base", "temporal"))
    trials = (
        kurversc.Trial(
            config=base,
            metric="roc_auc",
            validation_score=0.6,
            objective_score=0.6,
            feature_count=10,
            train_rows=80,
            validation_rows=20,
            feature_seconds=1.0,
            model_seconds=0.1,
        ),
        kurversc.Trial(
            config=temporal,
            metric="roc_auc",
            validation_score=0.7,
            objective_score=0.7,
            feature_count=20,
            train_rows=80,
            validation_rows=20,
            feature_seconds=2.0,
            model_seconds=0.1,
        ),
    )
    result = kurversc.FitResult(
        task="classification",
        metric="roc_auc",
        best_trial=trials[1],
        recommended_trial=trials[1],
        trials=trials,
        rerank_trials=trials,
        feature_audit=pd.DataFrame.from_records(
            [
                {
                    "table": "users",
                    "column": "segment",
                    "role": "categorical",
                    "eligible": True,
                    "structural": False,
                    "utility_score": 0.8,
                    "utility_rank": 1,
                    "selection_rank": 1,
                    "column_selection_strategy": "metadata_utility",
                    "eligible_families": "base,conditional",
                    "family_ranks": "base:1,conditional:1",
                    "within_family_budget": True,
                    "within_expanded_family_budget": True,
                    "eligible_budget_tiers": "4",
                    "reason": "eligible and utility-ranked",
                }
            ]
        ),
    )

    episodes = kurversc.episodes_from_fit_result(
        result,
        database_profile=_profile("database-a"),
        task_profile=_task(),
        runtime_profile=_runtime(),
    )

    assert [episode.stage for episode in episodes] == ["screening", "rerank"]
    assert episodes[0].stage_weight < episodes[1].stage_weight
    assert episodes[0].observations[1].is_search_best
    assert episodes[0].observations[1].plan["feature_families"] == [
        "base",
        "temporal",
    ]
    assert episodes[0].extras["column_selection_strategy"] == "metadata_utility"
    assert episodes[0].extras["column_selection"][0]["column"] == "segment"


def test_beam_pruned_candidate_is_censored_not_negative_supervision() -> None:
    trial = kurversc.Trial(
        config=kurversc.GraphConfig(feature_families=("base", "context")),
        metric="roc_auc",
        validation_score=float("nan"),
        objective_score=float("-inf"),
        feature_count=0,
        train_rows=0,
        validation_rows=0,
        feature_seconds=0.0,
        model_seconds=0.0,
        status="pruned",
        error="Candidate is not a one-family descendant of the top-2 beam",
    )

    observation = kurversc.CandidateObservation.from_trial(trial)

    assert observation.supervision == "censored"
    assert not observation.supervised


def test_trace_io_and_database_grouped_split(tmp_path) -> None:
    episodes = tuple(_episode(f"db-{index}") for index in range(6))
    path = tmp_path / "traces.jsonl"

    assert kurversc.write_episodes_jsonl(path, episodes) == 6
    restored = kurversc.read_episodes_jsonl(path)
    split = kurversc.split_episodes_by_database(
        restored,
        validation_databases=("db-4",),
        test_databases=("db-5",),
    )

    assert len(path.read_text(encoding="utf-8").splitlines()) == 6
    assert (
        json.loads(path.read_text(encoding="utf-8").splitlines()[0])["trace_version"]
        == "1.0"
    )
    assert split.validation_databases == ("db-4",)
    assert split.test_databases == ("db-5",)
    assert set(split.train_databases).isdisjoint(split.validation_databases)
    assert set(split.train_databases).isdisjoint(split.test_databases)


def test_automatic_final_holdout_covers_classification_and_regression() -> None:
    episodes = tuple(_episode(f"class-{index}") for index in range(4)) + (
        _episode("regression-only", task_type="regression"),
    )

    split = kurversc.split_episodes_by_database(
        episodes,
        test_size=2,
        validation_size=1,
        required_test_task_types=("classification", "regression"),
    )

    assert {episode.task_profile.task_type for episode in split.test} == {
        "classification",
        "regression",
    }


def test_candidate_frame_excludes_observed_costs_from_model_inputs() -> None:
    frame = candidate_frame((_episode("database-a"),))

    assert "plan__feature_families__has__temporal" in frame
    assert "feature_seconds" not in frame
    assert "feature_count" not in frame
    assert frame.loc[frame["_candidate_id"] == "deep", "_relevance"].item() == 0
    summary = summarize_episodes((_episode("database-a"),))
    assert summary["database_count"] == 1
    assert summary["candidate_supervision_counts"] == {"observed": 3}


def test_catboost_ranker_smoke_round_trip(tmp_path) -> None:
    training = tuple(_episode(f"train-{index}") for index in range(4))
    validation = (_episode("validation"),)
    ranker = KurvePlannerRanker.fit(
        training,
        validation_episodes=validation,
        config=PlannerTrainingConfig(iterations=8, depth=3, thread_count=1),
    )

    scores = ranker.score_episode(validation[0])
    evaluation = evaluate_ranker(ranker, validation)
    ranker.save(tmp_path)
    restored = KurvePlannerRanker.load(tmp_path)

    assert scores.shape == (3,)
    assert evaluation["summary"]["episode_count"] == 1
    np.testing.assert_allclose(scores, restored.score_episode(validation[0]))


def test_training_cli_dry_run_writes_auditable_split(tmp_path) -> None:
    episodes = tuple(_episode(f"class-{index}") for index in range(4)) + (
        _episode("regression-only", task_type="regression"),
    )
    traces = tmp_path / "traces.jsonl"
    output = tmp_path / "artifact"
    kurversc.write_episodes_jsonl(traces, episodes)

    exit_code = planner_main(
        [
            str(traces),
            "--output",
            str(output),
            "--test-size",
            "2",
            "--validation-size",
            "1",
            "--dry-run",
        ]
    )

    split = json.loads((output / "split.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(split["test_databases"]) == 2
    assert not (output / "ranker.cbm").exists()


def test_full_fidelity_collection_emits_all_explicit_candidates(tmp_path) -> None:
    users = pd.DataFrame(
        {
            "user_id": range(40),
            "segment": ["a" if value % 3 else "b" for value in range(40)],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": range(120),
            "user_id": [value // 3 for value in range(120)],
            "amount": [float(value % 11) for value in range(120)],
            "occurred_at": pd.date_range("2025-01-01", periods=120, freq="h"),
        }
    )
    labels = pd.DataFrame(
        {
            "user_id": range(40),
            "churn": [value % 2 for value in range(40)],
            "split": ["train"] * 30 + ["validation"] * 10,
        }
    )
    trace_path = tmp_path / "collection.jsonl"
    configs = (
        kurversc.GraphConfig(feature_families=("base",), auto_annotate_features=False),
        kurversc.GraphConfig(
            feature_families=("base", "temporal"), auto_annotate_features=False
        ),
    )

    collected = kurversc.collect_planner_search(
        kurversc.Table(
            users,
            # The search root must cover both the explicit training and
            # validation entities in the task-connected label cohort.
            search_source=pd.concat([users.head(8), users.tail(2)], ignore_index=True),
            name="users",
            key="user_id",
            prefix="user",
        ),
        kurversc.Labels(
            labels,
            key="user_id",
            target="churn",
            split="split",
            search_source=pd.concat(
                [labels.head(6), labels.tail(2)], ignore_index=True
            ),
        ),
        tables=(
            kurversc.Table(
                events,
                name="events",
                key="event_id",
                date="occurred_at",
                prefix="event",
            ),
        ),
        relationships=(
            kurversc.Relationship(
                parent="users",
                child="events",
                parent_key="user_id",
                child_key="user_id",
            ),
        ),
        database_profile=_profile("database-a"),
        task_profile=_task(),
        runtime_profile=_runtime(),
        trace_path=trace_path,
        append=False,
        graph_configs=configs,
        infer_ts_periods=True,
        model_params={"iterations": 8, "depth": 3},
        continue_on_error=False,
    )

    assert collected.fit_result.fitted_model is None
    assert len(collected.fit_result.trials) == 2
    assert {trial.stage for trial in collected.fit_result.trials} == {"collection"}
    assert all(trial.sample_rows is None for trial in collected.fit_result.trials)
    # Full-fidelity collection removes row caps but preserves the caller's
    # explicit task-connected search cohort.
    assert all(trial.train_rows == 6 for trial in collected.fit_result.trials)
    assert all(trial.validation_rows == 2 for trial in collected.fit_result.trials)
    assert collected.written_records == 1
    assert len(collected.episodes[0].observations) == 2
    assert all(
        len(observation.cutoff_scores) == 1
        for observation in collected.episodes[0].observations
    )
    assert kurversc.read_episodes_jsonl(trace_path) == collected.episodes
