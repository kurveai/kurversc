from __future__ import annotations

from pathlib import Path

import kurversc.checkpointing as checkpointing
from kurversc.checkpointing import SearchCheckpoint
from kurversc.search import GraphConfig, Trial


def completed_trial(config: GraphConfig, *, score: float = 0.75) -> Trial:
    return Trial(
        config=config,
        metric="roc_auc",
        validation_score=score,
        objective_score=score,
        feature_count=12,
        train_rows=100,
        validation_rows=50,
        feature_seconds=1.0,
        model_seconds=0.5,
        model=object(),
        feature_columns=("signal",),
        execution_plan={"records": [{"operation": "count"}]},
        stage="screening",
        sample_rows=None,
        cutoff_labels=("recent",),
        cutoff_scores=(score,),
    )


def test_search_checkpoint_round_trips_completed_trial(tmp_path: Path) -> None:
    config = GraphConfig(feature_families=("base", "state"), depth=2)
    checkpoint = SearchCheckpoint(tmp_path, contract={"dataset": "example-v1"})

    path = checkpoint.save(completed_trial(config))
    restored = checkpoint.load(config, stage="screening", sample_rows=None)

    assert path is not None and path.exists()
    assert restored is not None
    assert restored.validation_score == 0.75
    assert restored.execution_plan == {"records": [{"operation": "count"}]}
    assert restored.model is None


def test_search_checkpoint_isolates_changed_contracts(tmp_path: Path) -> None:
    config = GraphConfig()
    original = SearchCheckpoint(tmp_path, contract={"dataset": "example-v1"})
    changed = SearchCheckpoint(tmp_path, contract={"dataset": "example-v2"})
    original.save(completed_trial(config))

    assert original.contract_fingerprint != changed.contract_fingerprint
    assert changed.load(config, stage="screening", sample_rows=None) is None


def test_search_checkpoint_isolates_changed_runtimes(
    tmp_path: Path, monkeypatch
) -> None:
    config = GraphConfig()
    monkeypatch.setattr(
        checkpointing,
        "search_runtime_identity",
        lambda: {"graphreduce_version": "1.10.14"},
    )
    original = SearchCheckpoint(tmp_path, contract={"dataset": "example-v1"})
    original.save(completed_trial(config))
    monkeypatch.setattr(
        checkpointing,
        "search_runtime_identity",
        lambda: {"graphreduce_version": "1.10.17"},
    )
    changed = SearchCheckpoint(tmp_path, contract={"dataset": "example-v1"})

    assert original.contract_fingerprint != changed.contract_fingerprint
    assert changed.load(config, stage="screening", sample_rows=None) is None


def test_search_checkpoint_does_not_persist_failed_trial(tmp_path: Path) -> None:
    config = GraphConfig()
    checkpoint = SearchCheckpoint(tmp_path, contract={"dataset": "example-v1"})
    trial = completed_trial(config)
    trial.status = "failed"

    assert checkpoint.save(trial) is None
    assert checkpoint.load(config, stage="screening", sample_rows=None) is None
