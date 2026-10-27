import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.benchmark_recovered_readouts import matched_gate, phase_frames
from kurversc.signal_challengers import ChallengerUnavailable
from kurversc.temporal_policy import TemporalFitPolicy
from scripts.frozen_parent_semantic_embedding import FrozenParentSemanticEmbedding
from scripts.run_recovered_staged_cloud import CANARIES, recovery_manifest
from scripts.run_staged_complete_cloud import build_manifest
from scripts.run_default_relarena_cloud import task_pairs, source_files
from scripts.schema_semantic_recurrence import prepare_phase, scalar_payload


def write_table(path, frame, *, foreign, time_col, key=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({k.encode():json.dumps(v).encode() for k,v in
        dict(fkey_col_to_pkey_table=foreign, time_col=time_col, pkey_col=key).items()})
    pq.write_table(table, path)


def fixture(tmp_path):
    source = tmp_path / "source"
    ids = np.tile(np.arange(42), 4)
    times = np.repeat(pd.to_datetime(["2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01"]), 42)
    train = pd.DataFrame(dict(value=ids % 7,
        actors_Description=[f"long meaningful biography about science language actor unique{i}" for i in ids],
        label=ids % 2, __event_entity=ids, __event_cutoff=times))
    held = train.iloc[:42].copy().assign(__event_cutoff=pd.Timestamp("2020-12-01")).reset_index(drop=True)
    labels = train[["label", "__event_entity", "__event_cutoff"]].rename(
        columns={"__event_entity":"Actor", "__event_cutoff":"Date"})
    write_table(source / "tasks/problem/train.parquet", labels, foreign={"Actor":"actors"}, time_col="Date")
    write_table(source / "tasks/problem/val.parquet", labels.iloc[:0], foreign={"Actor":"actors"}, time_col="Date")
    events = pd.DataFrame(dict(Actor=np.tile(np.arange(42), 3),
        Date=np.repeat(pd.to_datetime(["2020-06-01", "2020-11-01", "2021-01-01"]), 42)))
    write_table(source / "db/events.parquet", events, foreign={"Actor":"actors"}, time_col="Date")
    meta = dict(task="problem", target="label", metric="roc_auc", label_horizon="30d", validation_cutoffs=["2020-12-01"],
        feature_schema=dict(columns=["value", "actors_Description"],
                            categorical_columns=["actors_Description"], datetime_columns=[]))
    return source, train, held, meta


def backend(documents):
    return np.asarray([np.frombuffer(hashlib.sha256(d.encode()).digest(), dtype=np.uint8)[:8] / 255
                       for d in documents], dtype="float32")


def test_semantic_phase_uses_declared_schema_and_mature_entity_disjoint_labels(tmp_path):
    source, train, held, meta = fixture(tmp_path)
    phase = prepare_phase(train, held, metadata=meta, source_root=source, origin="2020-12-01",
        output=tmp_path / "phase", workers=1,
        encoder=FrozenParentSemanticEmbedding(dimensions=4, backend=backend),
        teacher_params=dict(iterations=3, depth=2, thread_count=1))
    assert len(phase["oof"]) == len(train)
    assert np.isfinite(phase["oof"]).all()
    assert not {"__event_entity", "__event_cutoff", "label"} & set(phase["x"])
    assert phase["correction"].shape[0] == len(train)
    assert "semantic_propensity_crossfit_logit" in phase["correction"]
    with pytest.raises(ValueError, match="Immature"):
        prepare_phase(train.assign(__event_cutoff=pd.Timestamp("2020-12-01")), held,
                      metadata=meta, source_root=source, origin="2020-12-01", output=tmp_path / "bad")


def test_later_forecasts_cannot_expose_events_after_authorized_phase(tmp_path):
    source, train, held, meta = fixture(tmp_path)
    late = held.assign(__event_cutoff=pd.Timestamp("2021-02-01"))
    _, (training, future) = scalar_payload(train, late, source_root=source,
        target=("problem", "label"), horizon="30d", origin="2020-12-01", output=tmp_path / "state")
    # Jan1 events are after the source boundary and must not appear even for
    # February forecasts. June/November events remain the only visible rows.
    np.testing.assert_array_equal(future.tl_events_730, np.full(42, 2))
    assert len(training) == len(train)
    assert future.ls_count.max() == 4


def test_complete_confident_gate_and_negative_older_fold_fallback():
    rows = []
    truth = np.tile([0., 1.], 20)
    for origin in ("2020-01-01", "2020-02-01", "2020-03-01"):
        for seed in (0, 1, 2):
            rows.append(dict(origin=origin, seed=seed, truth=truth, groups=np.arange(len(truth)),
                             control=truth + 1, vectors={"good":truth + .5}))
    gate = matched_gate(rows, metric="mae", recipes=["good"], allowance=200)
    assert gate["selected"] == "good"
    assert matched_gate(rows[:-1], metric="mae", recipes=["good"], allowance=200)["selected"] == "incumbent"
    rows[0] = {**rows[0], "vectors":{"good":truth + 2}}
    assert matched_gate(rows, metric="mae", recipes=["good"], allowance=200)["selected"] == "incumbent"
    with pytest.raises(ValueError, match="Duplicate"):
        matched_gate(rows + rows[:1], metric="mae", recipes=["good"], allowance=200)


def test_suite_queues_all21_but_requires_both_recovery_canaries(tmp_path):
    original = build_manifest(tmp_path, task_pairs(), [], max_workers=5, cpus=64)
    current = recovery_manifest(original)
    by_name = {j["name"]:j for j in current["jobs"]}
    gate = by_name["readout-recovery-gate"]
    assert {f"{d}_{t}" for d,t in CANARIES} <= set(gate["depends"])
    for dataset, task in task_pairs():
        job = by_name[f"{dataset}_{task}"]
        assert "scripts.run_recovered_staged_cloud" in job["command"]
        assert ("readout-recovery-gate" in job["depends"]) == ((dataset,task) not in CANARIES)
    # Do not mutate the existing profile or its frozen manifest.
    assert any(j["name"] == "attendance-recovery-gate" for j in original["jobs"])


def test_optional_capacity_is_checked_before_state_sql_or_weight_loading(tmp_path, monkeypatch):
    source, train, held, meta = fixture(tmp_path)
    monkeypatch.setattr("scripts.schema_semantic_recurrence.scalar_payload",
                        lambda *a, **kw: pytest.fail("Capacity must precede source SQL"))
    for options, reason in ((dict(maximum_rows=len(train)-1), "rows"),
                            (dict(maximum_documents=2), "documents")):
        with pytest.raises(ChallengerUnavailable, match=reason):
            prepare_phase(train, held, metadata=meta, source_root=source, origin="2020-12-01",
                          output=tmp_path / reason, **options)


def test_large_payload_preserves_labels_and_baseline_without_reading_features(tmp_path, monkeypatch):
    _, train, held, meta = fixture(tmp_path)
    cache, snapshots = tmp_path / "cache", tmp_path / "snapshots"
    cache.mkdir()
    folder = snapshots / "fold0"
    folder.mkdir(parents=True)
    train.to_parquet(cache / "history.parquet", index=False)
    held.to_parquet(folder / "validation.parquet", index=False)
    (folder / "manifest.json").write_text(json.dumps(dict(snapshot_cutoff="2020-12-01",
        latest_observed_label_complete_at="2020-11-01", origin="2020-12-01")))
    monkeypatch.setattr("scripts.benchmark_recovered_readouts.cache_locations", lambda root: (cache, snapshots))
    monkeypatch.setattr("scripts.benchmark_recovered_readouts.policy_support",
                        lambda *a, **kw: dict(cutoffs=train.__event_cutoff.unique()))
    history, forecast, phase = phase_frames(tmp_path, dict(metadata=meta, folds=[{}]),
        TemporalFitPolicy("incumbent"), 0, maximum_feature_cells=1)
    assert list(history) == ["label", "__event_entity", "__event_cutoff"]
    assert list(forecast) == list(history)
    assert not phase["optional_fit_available"]
    np.testing.assert_array_equal(history.label, train.label)
    with pytest.raises(ChallengerUnavailable, match="cells"):
        phase_frames(tmp_path, dict(metadata=meta), TemporalFitPolicy("incumbent"),
                     0, frames=3, maximum_feature_cells=1)
    with pytest.raises(ChallengerUnavailable, match="rows"):
        phase_frames(tmp_path, dict(metadata=meta), TemporalFitPolicy("incumbent"),
                     0, frames=3, maximum_rows=1)


def test_two_available_origins_are_valid_but_single_origin_cannot_promote():
    truth = np.tile([0., 1.], 20)
    rows = [dict(origin=o, seed=s, truth=truth, groups=np.arange(len(truth)),
                 control=truth + 1, vectors={"good":truth + .5})
            for o in ("2020-01-01", "2020-02-01") for s in (0, 1, 2)]
    assert matched_gate(rows, metric="mae", recipes=["good"], allowance=200)["selected"] == "good"
    assert matched_gate(rows[:3], metric="mae", recipes=["good"], allowance=200)["selected"] == "incumbent"


def test_source_freeze_includes_required_runtime_identity_asset():
    from relarena.identity import CHECKSUMS_PATH
    assert CHECKSUMS_PATH.resolve() in source_files()
