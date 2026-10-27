import json

import pandas as pd
import pytest

from kurversc.temporal_policy import bounded_temporal_policies, policy_support
from scripts.benchmark_supported_official import selected_policy_evidence
from scripts.run_temporal_policy_campaign import (
    campaign_tasks,
    remaining_seconds,
    retain_incumbent,
)
from scripts.temporal_campaign_support import (
    required_policy_cutoffs,
    retrospective_folds,
)


def test_folds_are_disjoint_recent_and_adapt_to_short_history():
    dates = pd.date_range("2020-01-01", periods=10)
    folds = retrospective_folds(dates, size=2, origins=3)
    assert [[pd.Timestamp(cut) for cut in f] for f in folds] == [
        list(dates[4:6]),
        list(dates[6:8]),
        list(dates[8:10]),
    ]
    assert len(retrospective_folds(dates[:7], size=2)) == 2
    assert retrospective_folds(dates[:6], size=2) == []


def test_support_union_preserves_timestamp_types_and_every_menu_cohort():
    dates = pd.date_range("2020-01-01", periods=80)
    training = pd.DataFrame({"date": dates[:-2].repeat(100), "label": 0})
    validation = pd.DataFrame({"date": dates[-2:].repeat(100), "label": 1})
    kwargs = dict(timestamp="date", horizon="1d", boundary_lag="2d", task="regression")
    required, folds = required_policy_cutoffs(training, validation, **kwargs)
    assert all(isinstance(cut, pd.Timestamp) for cut in required)
    assert 0 < len(required) < 40
    labels = pd.concat([training, validation])
    for fold in folds:
        origin = pd.Timestamp(fold[0]) - pd.Timedelta("2d")
        for policy in bounded_temporal_policies("regression"):
            support = policy_support(
                labels,
                timestamp="date",
                origin=origin,
                label_horizon="1d",
                refit_frames=2,
                policy=policy,
            )
            assert set(map(pd.Timestamp, support["cutoffs"])) <= set(required)
            assert all(
                pd.Timestamp(cut) + pd.Timedelta("1d") <= origin
                for cut in support["cutoffs"]
            )
    changed = training.copy()
    changed["label"] = range(len(changed))
    repeated, _ = required_policy_cutoffs(changed, validation, **kwargs)
    assert repeated == required


def test_total_task_budget_is_shared_across_stages():
    assert (
        remaining_seconds(
            {"cache": {"elapsed_seconds": 7200}, "audit": {"elapsed_seconds": 3600}}, 4
        )
        == 3600
    )
    assert remaining_seconds({"official": {"elapsed_seconds": 5000}}, 1) == 0


def test_campaign_has_21_distinct_tasks_not_score_selected(tmp_path):
    rows = [
        {"dataset": "example", "task": f"target{i}", "test_score": i} for i in range(21)
    ]
    path = tmp_path / "tasks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    expected = campaign_tasks(path)
    pd.DataFrame([{**row, "test_score": -row["test_score"]} for row in rows]).to_csv(
        path, index=False
    )
    assert campaign_tasks(path) == expected
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly 21"):
        campaign_tasks(path)


def test_short_history_retains_incumbent_without_challenger_evidence(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "selection"
    cache.mkdir()
    metadata = dict(
        dataset="example", task="target", validation_cutoffs=26, checkpoint_sha256="abc"
    )
    (cache / "materialization.json").write_text(json.dumps(metadata))
    (cache / "frames.json").write_text(json.dumps(dict(control_score=0.7)))
    retain_incumbent(cache, output)
    evidence, arm = selected_policy_evidence(output / "results.json")
    assert arm == "incumbent"
    assert evidence["records"][0]["score"] == 0.7
    assert evidence["selected_policy"]["training_frames"] == 1
    assert evidence["selection_audit"]["reason"].startswith("insufficient_history")
