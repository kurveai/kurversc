import numpy as np
import pandas as pd
import pytest

import kurversc.core as core
from kurversc.modeling import FrameEnsemble
from scripts.count_median_faithful_inputs import capture_sampled_inputs, canonical_vectors, selected_members


def test_actual_sampled_calls_and_final_member_identity_are_preserved(tmp_path, monkeypatch):
    class Model:
        pass

    def fit(*args, **kwargs):
        return Model(), .1, ()

    monkeypatch.setattr(core, "_fit_temporal_frame_estimator", fit)
    x = pd.DataFrame(dict(past=[9., 3., 8.]), index=[44, 2, 10])
    y = pd.Series([0, 1, 4], index=x.index)
    kwargs = dict(model_backend="catboost", task="regression", categorical=[], random_state=0,
                  model_params=dict(iterations=1))
    with capture_sampled_inputs(tmp_path) as (records, identities):
        a = core._fit_temporal_frame_estimator(x, y, **kwargs)[0]
        b = core._fit_temporal_frame_estimator(x.iloc[::-1], y.iloc[::-1], **kwargs)[0]
        assert selected_members(FrameEnsemble([b, a], task="regression"), identities) == [1, 0]
        np.testing.assert_array_equal(pd.read_parquet(tmp_path / "member0/features.parquet").past, x.past)
        np.testing.assert_array_equal(pd.read_parquet(tmp_path / "member1/labels.parquet").target, y.iloc[::-1])
        assert records[0]["rows"] == 3
    assert core._fit_temporal_frame_estimator is fit
    with pytest.raises(ValueError, match="map one-to-one"):
        selected_members(FrameEnsemble([Model()], task="regression"), identities)


def test_masked_forecasts_restore_original_order_and_reject_duplicates():
    frames = [pd.DataFrame(dict(__query_order=[2, 0], __event_entity=[30, 10], __event_cutoff="2020-01-01")),
              pd.DataFrame(dict(__query_order=[1], __event_entity=[20], __event_cutoff="2020-02-01"))]
    keys, vectors = canonical_vectors(frames, dict(head=[np.array([.3, .1]), np.array([.2])]), expected_rows=3)
    np.testing.assert_array_equal(keys.__event_entity, [10, 20, 30])
    np.testing.assert_array_equal(vectors["head"], [.1, .2, .3])
    frames[1]["__query_order"] = 0
    with pytest.raises(ValueError, match="exactly once"):
        canonical_vectors(frames, dict(head=[np.array([.3, .1]), np.array([.2])]), expected_rows=3)
