from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from kurversc.signal_challengers import BoundedTextReadout, SignalChallenger, mature_support_mask
from kurversc.temporal_policy import TemporalFitPolicy
from scripts import benchmark_signal_challengers as runner
from scripts.schema_exposure_challenger import ExposurePath, SchemaExposureQuality


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_text_projection_preserves_mature_population_and_exact_predictions(task, monkeypatch):
    n = 160
    x = pd.DataFrame({"parent_text": [f"A sufficiently long descriptive document about shared topic {i % 40} and group {i % 2}." for i in range(n)],
        "child_text": [f"A different long description for child record number {i % 40}." for i in range(n)],
        "numeric_payload": np.arange(n, dtype=float)}, index=np.arange(n)*3+7)
    y = pd.Series(np.arange(n) % 2, index=x.index)
    if task == "regression":
        y = y.astype(float)*3 + np.arange(n)/100
    times = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=4, freq="7D"), 40), index=x.index)
    metadata = dict(metric="roc_auc" if task == "classification" else "mae", label_horizon="1D", validation_cutoffs=1)
    base = TemporalFitPolicy("incumbent")
    origin = pd.Timestamp("2024-01-25")
    mask, support = mature_support_mask(times, origin=origin, horizon="1D", refit_frames=1, policy=base)
    categorical = ["parent_text", "child_text"]
    reference = BoundedTextReadout(task=task, random_state=0).fit(x.loc[mask], y.loc[mask],
        categorical=categorical, root_prefix="parent")
    query = x.iloc[:12]
    expected = reference.predict_values(query)
    original = x.copy(deep=True)
    before = BoundedTextReadout.fit
    observed = []

    def bounded_fit(self, features, labels, **kwargs):
        assert list(features) == categorical
        pd.testing.assert_frame_equal(features, x.loc[mask, categorical])
        pd.testing.assert_series_equal(labels, y.loc[mask])
        observed.append(len(features))
        return before(self, features, labels, **kwargs)

    monkeypatch.setattr(BoundedTextReadout, "fit", bounded_fit)
    prediction, audit = runner.fit_candidate(SignalChallenger("text", "text"), base, x, y, times, query,
        origin=origin, metadata=metadata, categorical=[0, 1], root="parent", seed=0, threads=1)
    np.testing.assert_array_equal(prediction, expected)
    assert audit == {**reference.audit, "support": support}
    assert observed == [int(mask.sum())]
    pd.testing.assert_frame_equal(x, original)


def test_routing_headers_keep_order_and_own_their_numeric_storage():
    frame = pd.DataFrame({"numeric_payload": [1., 2.], "target": [0., 1.],
        "__event_entity": [17, 21], "__event_cutoff": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "__position": [8, 3], "__order": [1, 0]})
    original = frame.copy(deep=True)
    result = runner.routing_headers(frame, dict(columns=["numeric_payload"]))
    pd.testing.assert_frame_equal(result, original.drop(columns="numeric_payload"))
    assert not np.shares_memory(result.target.to_numpy(), frame.target.to_numpy())
    result.iloc[0, result.columns.get_loc("target")] = -1
    pd.testing.assert_frame_equal(frame, original)


def test_exposure_groups_only_queries_and_preserves_original_index(monkeypatch):
    frame = pd.DataFrame({"numeric_payload": [1., 2., 3., 4.], "__event_entity": [10, 20, 30, 40],
        "__event_cutoff": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"])}, index=[9, 2, 7, 5])
    groupby = pd.DataFrame.groupby

    def queries_only(self, *args, **kwargs):
        assert list(self) == ["__event_cutoff", "__event_entity"]
        return groupby(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", queries_only)

    class Provider:
        last_audit = {"point_in_time": True}

        def transform(self, entities, cutoff):
            return dict(activity=pd.DataFrame({"a": entities.to_numpy()}),
                quality=pd.DataFrame({"b": entities.to_numpy()*2}))

    model = object.__new__(SchemaExposureQuality)
    model.provider = Provider()
    model.path = ExposurePath("sessions", "person", "events", "session", "items", "item", "response")
    result = model.transform(frame, phase_boundary=pd.Timestamp("2024-01-02"))
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [10, 20, 30, 40], "b": [20, 40, 60, 80]}, index=frame.index))
    assert model.last_audit["schema_path"] == asdict(model.path)
