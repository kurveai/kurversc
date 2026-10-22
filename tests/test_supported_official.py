import pytest
import json
from dataclasses import asdict

from kurversc.temporal_policy import TemporalFitPolicy
from scripts.benchmark_supported_official import (
    support_budget,
    selected_policy_evidence,
)


def test_floor_policy_transfers_row_floor_not_observed_frame_count():
    evidence = {"plans": {"floor1000": {"minimum_rows": 1000, "cutoffs": ["old"] * 47}}}
    assert support_budget(evidence, "floor1000") == (1000, 1, 64, False)


def test_fixed_frame_and_readout_policies():
    assert support_budget({"plans": {"pooled12": {}}}, "pooled12") == (1, 12, 12, False)
    assert support_budget({"support": "pooled12"}, "ordered_median") == (
        1,
        12,
        12,
        True,
    )
    assert support_budget({"support": "latest1 verified cache"}, "control") == (
        1,
        1,
        1,
        False,
    )


def test_unsupported_weighted_or_supplemented_policies_rejected():
    with pytest.raises(ValueError, match="unweighted"):
        support_budget({"plans": {}}, "recency_support")
    with pytest.raises(ValueError, match="Only direct"):
        support_budget({"support": "pooled12"}, "numeric")


def test_selected_policy_budget_is_replayed_without_manual_arm_choices():
    policy = TemporalFitPolicy(
        "ordered12", mode="pooled", readout="ordered_median", training_frames=12
    )
    assert support_budget({"selected_policy": asdict(policy)}, "ordered12") == (
        1,
        12,
        64,
        True,
    )
    with pytest.raises(ValueError, match="validation-selected"):
        support_budget({"selected_policy": asdict(policy)}, "floor1000")


def test_selector_artifact_requires_an_eligible_challenger_and_phase_snapshots(
    tmp_path,
):
    path = tmp_path / "results.json"
    policy = asdict(TemporalFitPolicy("pooled12", mode="pooled", training_frames=12))
    audit = dict(
        dataset="example",
        task="target",
        selected_policy=policy,
        decision=dict(
            selected="pooled12", candidates={"pooled12": {"reason": "eligible"}}
        ),
        records=[dict(policy="pooled12", origin="2020-01-01", seed=0, score=0.7)],
    )
    manifest = dict(
        metadata=dict(
            dataset="example",
            task="target",
            validation_cutoffs=2,
            checkpoint_sha256="abc",
        )
    )
    path.write_text(json.dumps(audit))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="phase-frozen"):
        selected_policy_evidence(path)
    manifest["phase_frozen_cache"] = "snapshot-cache"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    evidence, arm = selected_policy_evidence(path)
    assert arm == "pooled12" and evidence["records"][0]["arm"] == arm
    audit["decision"]["candidates"]["pooled12"]["reason"] = "older_origin_regression"
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="not admitted"):
        selected_policy_evidence(path)
