"""Exercise the complete opt-in cache CLI with fresh in-memory fit fixtures.

No external checkpoint is deserialized: the trial is created by this test's
GraphReduce fit, and the checkpoint stub is used only for identity hashing.
"""

from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd

import kurversc
from kurversc.relbench import RelBenchProblem
from scripts import benchmark_sparse_training as benchmark


def test_complete_materialization_and_resume_preserve_all_full_frames(tmp_path, monkeypatch):
    ids = np.arange(12)
    parents = kurversc.Table(pd.DataFrame(dict(id=ids, value=ids.astype(float))),
        name="actors", key="id", timeless=True, prefix="actor")
    children = kurversc.Table(pd.DataFrame(dict(id=ids, when=pd.Timestamp("2019-12-01"), amount=ids**2)),
        name="events", date="when", prefix="event")
    clocks = list(pd.date_range("2020-01-01", periods=4, freq="2D"))
    labels = pd.concat([pd.DataFrame(dict(id=ids, at=clock, target=ids%2,
        split="validation" if i==3 else "train")) for i, clock in enumerate(clocks)], ignore_index=True)
    task = SimpleNamespace(entity_table="actors", entity_col="id", time_col="at", target_col="target",
        task_type="binary_classification", timedelta=pd.Timedelta("2 days"))
    problem = RelBenchProblem("fixture", "binary", parents,
        kurversc.Labels(labels, key="id", timestamp="at", target="target", split="split", target_history=False),
        (children,), (kurversc.Relationship("actors", "events", "id", "id"),), "at",
        tuple(clocks[:3]), (clocks[3],), (), task)
    config = kurversc.GraphConfig(feature_families=("base",), auto_annotate_features=False,
        include_target_history=False, base_distribution_features=True)
    runtime = dict(full_training_frames=1, full_validation_frames=None,
        temporal_estimator_mode="frame_ensemble", estimator_train_rows=25,
        infer_ts_periods=False, model_params=dict(iterations=3, thread_count=1),
        duckdb_memory_limit="1GB", duckdb_max_temp_directory_size="1GB")
    initial = kurversc.fit(**problem.fit_kwargs(), **runtime, preselected_config=config, random_state=0)
    trial = SimpleNamespace(config=config, execution_plan=initial.execution_plan)
    checkpoint = tmp_path / "identity-only-not-a-pickle.txt"
    checkpoint.write_text("Fresh test-owned execution plan lives only in memory\n")
    train_table = SimpleNamespace(df=labels.loc[labels.split.eq("train")].copy())
    val_table = SimpleNamespace(df=labels.loc[labels.split.eq("validation")].copy())
    inner = SimpleNamespace(cutoff=clocks[2], train_table=train_table, eval_table=val_table,
        db_state=SimpleNamespace(table_dict={"actors": SimpleNamespace(df=parents.source, pkey_col="id", time_col=None)}))
    source = SimpleNamespace(task=task, inner_split=lambda: inner, run_identity=lambda: {})

    class System:
        def __init__(self, **kwargs):
            pass

        def _fit_arm(self, *args, **kwargs):
            selection = kwargs["selection"]
            return kurversc.fit(**problem.fit_kwargs(), **runtime, preselected_config=selection.config,
                preselected_execution_plan=selection.execution_plan, random_state=0), problem

    monkeypatch.setattr(benchmark, "_checkpoint", lambda *args: (checkpoint, trial))
    monkeypatch.setattr(benchmark, "RelBenchDatasetTask", lambda *args: source)
    monkeypatch.setattr(benchmark, "_fold_tables", lambda *args: (train_table, val_table))
    monkeypatch.setattr(benchmark, "AuditWinnerSystem", System)
    monkeypatch.setattr(benchmark, "resolve_cache_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, "_runtime_config", lambda: dict(runtime))
    monkeypatch.setattr(benchmark, "KURVERSC_DEFAULTS", {})

    for mode in ("memory", "stream"):
        command = ["benchmark", "--dataset", "fixture", "--task", "binary", "--fold", "current",
            "--output", str(tmp_path / mode), "--materialize-only", "--row-floors", "20",
            "--frame-counts", "3"]
        if mode == "stream":
            command.append("--stream-feature-cache")
        monkeypatch.setattr(sys, "argv", command)
        benchmark.main()
    for name in ("history", "validation"):
        expected = pd.read_parquet(tmp_path / "memory" / f"{name}.parquet")
        actual = pd.read_parquet(tmp_path / "stream" / f"{name}.parquet")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_categorical=False, check_exact=True)
    report = json.loads((tmp_path / "stream/materialization.json").read_text())
    assert report["history_rows"] == 36 and report["validation_rows"] == 12
    assert report["streamed_feature_capture"] is True
    assert report["control_score"] == initial.full_validation_score
    # Resume reads only the small alignment headers, not full payloads.
    benchmark.main()
    resumed = json.loads((tmp_path / "stream/materialization.json").read_text())
    for key in ("history_rows", "validation_rows", "control_score", "feature_schema"):
        assert resumed[key] == report[key]
