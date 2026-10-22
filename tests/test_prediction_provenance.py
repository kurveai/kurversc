import numpy as np
import pandas as pd
import pytest

from kurversc.prediction_provenance import (
    align_prediction, artifact_keys, key_arrays, key_fingerprint, masked_query_keys, query_positions,
)


def keys():
    return pd.DataFrame(dict(entity=[2, 5, 8], timestamp=pd.to_datetime(["2020-01-01"] * 3)))


def test_each_vector_uses_its_own_keys_and_queries_can_change_order():
    source = keys().iloc[[2, 0, 1]].reset_index(drop=True)
    destination = keys().iloc[[1, 2, 0]].reset_index(drop=True)
    out = align_prediction([.8, .2, .5], source_keys=source, query_keys=destination)
    np.testing.assert_array_equal(out, [.5, .8, .2])
    assert key_fingerprint(source) != key_fingerprint(destination)
    a = dict(groups=source.entity.to_numpy(), forecast_cutoffs=source.timestamp.to_numpy())
    pd.testing.assert_frame_equal(artifact_keys(a), source)


def test_missing_duplicate_changed_population_and_hidden_targets_fail_closed():
    with pytest.raises(ValueError, match="Unkeyed"):
        artifact_keys(dict(candidate=np.zeros(3)))
    with pytest.raises(ValueError, match="Unique"):
        query_positions(keys(), pd.concat([keys().iloc[:1]] * 3, ignore_index=True))
    with pytest.raises(ValueError, match="populations"):
        query_positions(keys(), keys().assign(entity=[2, 5, 9]))
    with pytest.raises(ValueError, match="hidden"):
        masked_query_keys(keys().assign(target=0), entity="entity", timestamp="timestamp", target="target")
    with pytest.raises(ValueError, match="finite"):
        align_prediction([1, np.nan, 0], source_keys=keys(), query_keys=keys())


@pytest.mark.parametrize("values", [pd.Series([2, 5, 8], dtype="Int64"), pd.Series(["a", "b", "c"], dtype="string")])
def test_nullable_keys_save_without_unsafe_object_arrays(tmp_path, values):
    frame = keys().assign(entity=values)
    k = masked_query_keys(frame, entity="entity", timestamp="timestamp")
    path = tmp_path / "keys.npz"
    np.savez_compressed(path, **key_arrays(k))
    with np.load(path, allow_pickle=False) as a:
        pd.testing.assert_frame_equal(artifact_keys(a), k)
