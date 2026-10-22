from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kurversc.task_context import ContextDatabase, ContextTable, PredictionTask, TaskContext
from kurversc.task_source import ContextTaskSource, CONTEXT_ENV, phase_source_root
from kurversc import protocol_runner


def splits():
    roots = ContextTable(pd.DataFrame(dict(id=[42, 7], segment=[2., 1.])), "id", {}, None, None, None)
    events = ContextTable(pd.DataFrame(dict(eid=[9, 3], uid=[7, 42], value=[1., 2.],
        at=pd.to_datetime(["2020-01-01", "2020-02-01"]))), "eid", dict(uid="actors"), "at", None, None)
    db = ContextDatabase(dict(actors=roots, activity=events), exclude_structural_aliases=False)
    labels = ContextTable(pd.DataFrame(dict(actor=[42, 7], when=pd.to_datetime(["2020-01-01"]*2),
        answer=[1, 0])), None, dict(actor="actors"), "when", None, None)
    val = ContextTable(labels.df.assign(when=pd.Timestamp("2020-03-01")), None,
        dict(actor="actors"), "when", None, None)
    query = ContextTable(val.df.iloc[::-1].drop(columns="answer").assign(when=pd.Timestamp("2020-04-01")),
        None, dict(actor="actors"), "when", None, None)
    inner = SimpleNamespace(db_state=db, train_table=labels, eval_table=val, cutoff=pd.Timestamp("2020-03-01"))
    outer = SimpleNamespace(db_state=db, train_table=labels, val_table=val, eval_table=query,
        cutoff=pd.Timestamp("2020-04-01"))
    return inner, outer


def definition():
    return PredictionTask("actors", "actor", "when", "answer", "28d", name="new-problem")


def test_protocol_roundtrip_does_not_rename_sort_or_change_clock_metadata(tmp_path, monkeypatch):
    inner, outer = splits()
    context = TaskContext.from_protocol_splits(task=definition(), inner_split=inner,
        outer_split=outer, dataset_name="unregistered")
    assert isinstance(context, TaskContext)
    bundle = context.export(tmp_path / "bundle")
    source = ContextTaskSource(bundle)
    assert source.manifest["version"] == 2 and source.task.label_available_at is None
    for original, actual in ((inner, source.inner_split()), (outer, source.outer_split())):
        assert not actual.db_state.exclude_structural_aliases
        assert list(original.db_state.table_dict) == list(actual.db_state.table_dict)
        for name, expected in original.db_state.table_dict.items():
            table = actual.db_state.table_dict[name]
            pd.testing.assert_frame_equal(expected.df.reset_index(drop=True), table.df.reset_index(drop=True), check_exact=True)
            assert table.time_col == expected.time_col and table.available_col is None and table.cutoff_boundary is None
            assert table.fkey_col_to_pkey_table == expected.fkey_col_to_pkey_table
        for role in ("train_table", "eval_table", *(["val_table"] if original is outer else [])):
            pd.testing.assert_frame_equal(getattr(original, role).df.reset_index(drop=True),
                getattr(actual, role).df.reset_index(drop=True), check_exact=True)
    assert source.outer_split().eval_table.df.actor.tolist() == [7, 42]
    assert source.mature_labels(inner.train_table.df, "2020-01-28").empty
    assert len(source.mature_labels(inner.train_table.df, "2020-01-29")) == 2
    monkeypatch.setenv(CONTEXT_ENV, str(bundle))
    folder = phase_source_root("unregistered")
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(folder / "db/activity.parquet").schema_arrow.metadata
    assert json.loads(meta[b"available_at"]) is None
    assert json.loads(meta[b"cutoff_boundary"]) is None


def test_protocol_export_rejects_hidden_target_columns_and_uncensored_sources():
    inner, outer = splits()
    outer.eval_table.df["answer"] = np.nan
    with pytest.raises(ValueError, match="held-out target"):
        TaskContext.from_protocol_splits(task=definition(), inner_split=inner, outer_split=outer)
    inner, outer = splits()
    inner.db_state.table_dict["activity"].df.loc[0, "at"] = pd.Timestamp("2022-01-01")
    with pytest.raises(ValueError, match="beyond"):
        TaskContext.from_protocol_splits(task=definition(), inner_split=inner, outer_split=outer)


def test_registered_bridge_calls_same_public_pipeline_and_can_reuse_owned_bundle(tmp_path, monkeypatch):
    inner, outer = splits()
    task = SimpleNamespace(task_type="binary_classification", entity_table="actors", entity_col="actor",
        time_col="when", target_col="answer", timedelta=pd.Timedelta("28d"))
    @dataclass
    class Identity:
        dataset: str = "unregistered"
        dataset_fingerprint: str = "caller_content_checksum"
        task_fingerprint: str = "task_definition_checksum"
    calls = []
    def run(context, **kwargs):
        loaded = ContextTaskSource(context)
        assert loaded.outer_split().eval_table.df.actor.tolist() == [7, 42]
        assert loaded.task.target_col not in loaded.outer_split().eval_table.df
        calls.append((context, kwargs))
        return SimpleNamespace(output=kwargs["output"])
    monkeypatch.setattr(protocol_runner, "run_staged_task", run)
    monkeypatch.setattr(protocol_runner, "staged_source_identity", lambda _: dict(code="fixed"))
    kwargs = dict(inner_split=inner, outer_split=outer, dataset_name="unregistered", task_name="new-problem",
        cache_directory=tmp_path, run_identity=Identity())
    first = protocol_runner.run_protocol_task(task, **kwargs)
    again = protocol_runner.run_protocol_task(task, **kwargs)
    assert first.output == again.output and calls[0][0] == calls[1][0]
    assert calls[0][1]["task_hours"] == 23.75
    with pytest.raises(ValueError, match="not silently ignored"):
        protocol_runner.run_protocol_task(task, seed=1, **kwargs)


def test_internal_graph_system_stays_native_and_does_not_reenter_complete_pipeline(monkeypatch):
    from scripts.run_kurversc_selected_relarena import AuditWinnerSystem
    from kurversc.staged_adapter import KurveRSCSystem
    called = []
    monkeypatch.setattr(KurveRSCSystem, "_run_native", lambda *a, **kw: called.append(kw) or np.array([.2]), raising=False)
    system = object.__new__(AuditWinnerSystem)
    prediction = system.run(object(), inner_split=object(), outer_split=object(), seed=0)
    np.testing.assert_array_equal(prediction, [.2])
    assert len(called) == 1
