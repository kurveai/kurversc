from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.streaming_feature_cache import FrameSink, capture_to_disk
from kurversc import core
from kurversc.modeling import prepare_prediction_features


def test_streamed_concatenation_preserves_order_keys_and_model_visible_values(tmp_path):
    frames = [pd.DataFrame(dict(value=pd.Series([1, None], dtype="Int64"), kind=pd.Categorical(["a", "b"]),
        clock=pd.to_datetime(["2001-01-01", None]), __event_entity=pd.Series([2**63+1, 2**63+2], dtype="uint64"),
        target=[0, 1], __event_cutoff=["2001-01-01"]*2)),
        pd.DataFrame(dict(value=[2., 2.5], kind=pd.Categorical(["b", "c"]),
        clock=pd.to_datetime(["2001-02-01", "2001-02-02"]), __event_entity=pd.Series([2**63+3, 2**63+4], dtype="uint64"),
        target=[1, 0], __event_cutoff=["2001-02-01"]*2))]
    sink = FrameSink(tmp_path / "parts")
    for frame in frames:
        sink.append(frame, frame.__event_cutoff.iloc[0])
    report = sink.concatenate(tmp_path / "history.parquet", cutoffs=["2001-01-01", "2001-02-01"], batch_rows=1)
    expected = pd.concat(frames, ignore_index=True)
    actual = pd.read_parquet(tmp_path / "history.parquet")
    np.testing.assert_array_equal(actual.__event_entity, expected.__event_entity)
    np.testing.assert_array_equal(actual.target, expected.target)
    schema = dict(columns=["value", "kind", "clock"], categorical_columns={"kind"}, datetime_columns={"clock"})
    a, b = [prepare_prediction_features(f, **schema) for f in (actual, expected)]
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    assert report["rows"] == 4 and report["fitter_frames_unchanged"]
    assert sink.concatenate(tmp_path / "history.parquet", cutoffs=["2001-01-01", "2001-02-01"], batch_rows=1) == report
    with pytest.raises(ValueError, match="identity"):
        sink.concatenate(tmp_path / "history.parquet", cutoffs=["2001-01-01"])


def test_capture_returns_the_identical_frame_to_the_fitter_and_restores_hook(tmp_path, monkeypatch):
    original_frame = pd.DataFrame(dict(root_id=[1, 2], x=[3., 4.], target=[0, 1], ready=["2001-01-02"]*2))
    called = []

    def original(*args, **kwargs):
        called.append(1)
        return original_frame

    monkeypatch.setattr(core, "_materialize_at_cutoff", original)
    spec = SimpleNamespace(available_at="ready")
    with capture_to_disk(["x"], "target", "id", "root", tmp_path / "parts") as sink:
        returned = core._materialize_at_cutoff(label_spec=spec, cut_date=pd.Timestamp("2001-01-01"))
        assert returned is original_frame
        assert len(sink.parts) == 1
        saved = pd.read_parquet(sink.parts[0]["path"])
        assert list(saved) == ["x", "target", "__kurversc_label_available_at", "__event_entity", "__event_cutoff"]
        assert list(saved.__event_entity) == [1, 2]
    assert core._materialize_at_cutoff is original and called == [1]


def test_stream_rejects_schema_changes_and_preserves_failed_parts(tmp_path):
    sink = FrameSink(tmp_path / "parts")
    sink.append(pd.DataFrame(dict(x=[1])), "2001-01-01")
    sink.append(pd.DataFrame(dict(other=[1])), "2001-02-01")
    with pytest.raises(ValueError, match="ordered"):
        sink.concatenate(tmp_path / "history.parquet", cutoffs=["2001-01-01", "2001-02-01"])
    assert len(list((tmp_path / "parts").glob("*.parquet"))) == 2


def test_restarted_capture_preserves_completed_parts_and_refuses_changed_values(tmp_path):
    frame = pd.DataFrame(dict(x=[1., 2.]))
    a = FrameSink(tmp_path / "parts")
    a.append(frame, "2001-01-01")
    previous = (tmp_path / "parts/frame0000.parquet").read_bytes()
    b = FrameSink(tmp_path / "parts")
    b.append(frame.copy(), "2001-01-01")
    assert previous == (tmp_path / "parts/frame0000.parquet").read_bytes()
    c = FrameSink(tmp_path / "parts")
    with pytest.raises(AssertionError):
        c.append(frame.assign(x=[9., 8.]), "2001-01-01")
    assert previous == (tmp_path / "parts/frame0000.parquet").read_bytes()


def test_real_graph_fit_refit_and_predictions_match_memory_capture(tmp_path):
    import kurversc
    from scripts.benchmark_frozen_event_summaries import _capture

    ids = np.arange(12)
    parents = kurversc.Table(pd.DataFrame(dict(id=ids, value=ids.astype(float))),
        name="actors", key="id", timeless=True, prefix="actor")
    children = kurversc.Table(pd.DataFrame(dict(id=ids, when=pd.Timestamp("2019-12-01"), amount=ids**2)),
        name="events", date="when", prefix="event")
    labels = pd.concat([pd.DataFrame(dict(id=ids, at=clock, target=ids%2,
        split="validation" if i==3 else "train")) for i,clock in enumerate(pd.date_range("2020-01-01", periods=4, freq="2D"))], ignore_index=True)
    config = kurversc.GraphConfig(feature_families=("base",), auto_annotate_features=False,
        include_target_history=False, base_distribution_features=True)
    options = dict(parent_node=parents, tables={"events":children},
        relationships=[kurversc.Relationship("actors", "events", "id", "id")],
        label_node=kurversc.Labels(labels, key="id", timestamp="at", target="target", split="split", target_history=False),
        preselected_config=config, full_training_frames=3, full_validation_frames=None,
        temporal_estimator_mode="pooled", estimator_train_rows=25,
        infer_ts_periods=False, model_params=dict(iterations=3, thread_count=1),
        duckdb_memory_limit="1GB", duckdb_max_temp_directory_size="1GB")
    first = kurversc.fit(**options)
    columns = first.fitted_model.feature_columns
    options["preselected_execution_plan"] = first.execution_plan
    with _capture(columns, "target", "id", "actors") as (frames, cuts):
        memory_fit = kurversc.fit(**options)
    with capture_to_disk(columns, "target", "id", "actors", tmp_path / "real-parts") as sink:
        disk_fit = kurversc.fit(**options)
    sink.concatenate(tmp_path / "all.parquet", cutoffs=cuts, batch_rows=3)
    expected = pd.concat(frames, ignore_index=True)
    actual = pd.read_parquet(tmp_path / "all.parquet")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_categorical=False, check_exact=True)
    schema = dict(columns=list(columns), categorical_columns=set(first.fitted_model.categorical_columns),
        datetime_columns=set(first.fitted_model.datetime_columns))
    x = prepare_prediction_features(actual, **schema)
    for member in ("estimator", "validation_estimator"):
        a, b = [getattr(result.fitted_model, member).predict_proba(x) for result in (memory_fit, disk_fit)]
        np.testing.assert_array_equal(a, b)
    assert memory_fit.full_validation_score == disk_fit.full_validation_score
    assert memory_fit.plan_fingerprint == disk_fit.plan_fingerprint
