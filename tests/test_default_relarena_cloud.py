from pathlib import Path

import pandas as pd
import pytest

from scripts.run_default_relarena_cloud import (
    build_manifest, clean_environment, ram_estimate, task_pairs, validate_complete_scores,
)
from scripts.recovered_reference_profile import (
    RecoveredReferenceSystem, execution_profile, recovered_reference_candidates,
)


def test_unique_official_21_task_grid():
    pairs = task_pairs()
    assert len(pairs) == len(set(pairs)) == 21
    assert set(dataset for dataset, _ in pairs) == {
        "rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm", "rel-stack", "rel-trial"}


def test_default_manifest_only_changes_execution_and_checkpoint_locations():
    manifest = build_manifest(Path("/fresh"), task_pairs(), [Path("/source.py")], 20, 64)
    jobs = [j for j in manifest["jobs"] if "/" in j.get("task", "")]
    assert len(jobs) == 21
    assert manifest["max_workers"] == 8
    assert manifest["reserve_gib"] == 128
    assert manifest["task_hours"] < 24
    for job in jobs:
        assert job["threads"] == 8
        assert "--profile" not in job["command"]
        assert "--signal-challengers" not in job["command"]
        assert "--search-workers" not in job["command"]
        assert job["depends"][0].startswith("prepare-rel-")
    elo = manifest["jobs"][-1]
    assert len(elo["depends"]) == 22
    assert "leaderboard-snapshot" in elo["depends"]


def test_admission_reservations_are_not_task_rss_caps():
    manifest = build_manifest(Path("/fresh"), task_pairs(), [], 8, 64)
    assert all("max_rss_gib" not in job for job in manifest["jobs"])
    assert ram_estimate("rel-hm", "user-churn") > 300
    assert ram_estimate("rel-stack", "user-badge") > ram_estimate("rel-f1", "driver-dnf")


def test_recovered_profile_is_uniform_and_not_the_regressed_native_recipe():
    system, config = execution_profile("recovered-reference")
    assert system is RecoveredReferenceSystem
    assert system.schema_depth == 3
    assert config["full_training_frames"] == config["search_training_frames"] == 1
    assert config["temporal_estimator_mode"] == "frame_ensemble"
    assert config["estimator_train_rows"] is None
    assert config["search_parallelism"] == 1
    candidates = recovered_reference_candidates()
    assert len(candidates) == len(set(candidates)) == 72
    assert all(c.categorical_low_cardinality_policy == "top_k" for c in candidates)
    assert all(not c.include_target_history and c.published_cutoff_compatibility for c in candidates)


def test_recovered_manifest_carries_the_same_profile_through_all_21_tasks():
    manifest = build_manifest(Path("/fresh"), task_pairs(), [], 8, 64,
        profile="recovered-reference")
    assert manifest["run_profile"] == "recovered-reference"
    for job in manifest["jobs"]:
        command = job["command"]
        assert command[command.index("--profile") + 1] == "recovered-reference"
        assert "--fixed-graph" not in command
    assert len([j for j in manifest["jobs"] if "/" in j.get("task", "")]) == 21


def test_recovered_arm_selects_then_replays_without_removing_observed_validation(
        monkeypatch, tmp_path):
    from types import SimpleNamespace
    import kurversc

    calls, adapters = [], []
    chosen = recovered_reference_candidates()[0]
    plan = {"records": [{"field": "signal"}]}
    monkeypatch.setenv("KURVERSC_SEARCH_CHECKPOINT_DIR", str(tmp_path))

    def problem(*args, **kwargs):
        adapters.append((args, kwargs))
        return SimpleNamespace(fit_kwargs=lambda: {})

    def fit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(recommended_config=chosen, execution_plan=plan)

    monkeypatch.setattr(kurversc, "relbench_problem_from_objects", problem)
    monkeypatch.setattr(kurversc, "fit", fit)
    system = RecoveredReferenceSystem()
    task = object()
    system._fit_arm(task, object(), object(), object(), seed=0, phase="inner")
    selection = SimpleNamespace(config=chosen, execution_plan=plan)
    system._fit_arm(task, object(), object(), object(), seed=0,
        phase="outer", selection=selection)
    assert "graph_configs" in calls[0] and "preselected_config" not in calls[0]
    assert calls[1]["preselected_config"] == chosen
    assert calls[1]["preselected_execution_plan"] == plan
    assert calls[1]["preselected_execution_plan"] is not plan
    assert "graph_configs" not in calls[1]
    assert [kw["max_train_timestamps"] for _, kw in adapters] == [3, 1]
    assert all(kw["include_target_history"] is False for _, kw in adapters)
    assert all(call["full_validation_frames"] is None for call in calls)


def test_research_environment_cannot_change_defaults(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("KURVERSC_"):
            monkeypatch.delenv(key)
    clean_environment()
    monkeypatch.setenv("KURVERSC_SEARCH_PROFILE", "staged")
    with pytest.raises(ValueError, match="inherited overrides"):
        clean_environment()


def test_full_elo_never_uses_previous_scores_or_partial_panels():
    pairs = task_pairs()
    frame = pd.DataFrame([dict(dataset=d, task=t, test_score=.5) for d, t in pairs])
    validate_complete_scores(pairs, frame)
    with pytest.raises(ValueError, match="21 fresh"):
        validate_complete_scores(pairs, frame.iloc[:-1])
    with pytest.raises(ValueError, match="21 fresh"):
        validate_complete_scores(pairs, pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, "test_score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_complete_scores(pairs, frame)
