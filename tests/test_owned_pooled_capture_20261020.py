from types import SimpleNamespace
import hashlib
import pickle

import numpy as np
import pandas as pd
import pytest

from kurversc import core
from scripts.capture_owned_pooled_native_20261020 import capture_frames, load_owned_trial


def test_ownership_and_checksum_precede_any_model_deserialization(tmp_path, monkeypatch):
    owned = tmp_path / "fresh"
    owned.mkdir()
    (owned / "source-identity.json").write_text("{}")
    outside = tmp_path / "untrusted.pkl"
    outside.write_bytes(b"never decode this file")
    called = []
    monkeypatch.setattr(pickle, "load", lambda *a: called.append(True))
    meta = dict(checkpoint=str(outside), checkpoint_sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="owned checkpoint"):
        load_owned_trial(owned, meta)
    p = owned / "incomplete.pkl"
    p.write_bytes(b"owned but changed")
    with pytest.raises(ValueError, match="checksum"):
        load_owned_trial(owned, dict(meta, checkpoint=str(p)))
    assert called == []


def test_new_test_owned_trial_requires_exact_frozen_schema(tmp_path):
    (tmp_path / "source-identity.json").write_text("{}")
    schema = dict(columns=["x"], categorical_columns=[], datetime_columns=[])
    trial = SimpleNamespace(status="completed", execution_plan=dict(kurversc_feature_schema=schema))
    p = tmp_path / "fresh-own-plan.pkl"
    p.write_bytes(pickle.dumps(dict(trial=trial)))
    meta = dict(checkpoint=str(p), checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), feature_schema=schema)
    assert load_owned_trial(tmp_path, meta)[1].execution_plan == trial.execution_plan
    with pytest.raises(ValueError, match="schema"):
        load_owned_trial(tmp_path, dict(meta, feature_schema=dict(schema, columns=["other"])))


def test_capture_separates_mature_rows_from_masked_forecasts_without_modifying_fitter_inputs(monkeypatch):
    train = pd.DataFrame(dict(id=[1, 2], x=[3., 4.], target=[0, 1]))
    query = pd.DataFrame(dict(id=[2, 1], x=[4., 3.], __kurversc_prediction_order__0=[0, 1]))
    current = [train]
    original = lambda *args, **kwargs: current[0]
    monkeypatch.setattr(core, "_materialize_at_cutoff", original)
    schema = dict(columns=["x"])
    with capture_frames(schema, target="target", entity="id", root="actors") as (training, forecasts):
        assert core._materialize_at_cutoff(cut_date="2001-01-01") is train
        core._materialize_at_cutoff(cut_date="2001-01-01")
        current[0] = query
        assert core._materialize_at_cutoff(cut_date="2001-02-01") is query
        core._materialize_at_cutoff(cut_date="2001-02-01")
        assert len(training) == len(forecasts) == 1
        assert "target" in next(iter(training.values()))
        assert "target" not in next(iter(forecasts.values()))
        np.testing.assert_array_equal(next(iter(forecasts.values())).__order, [0, 1])
    assert core._materialize_at_cutoff is original
