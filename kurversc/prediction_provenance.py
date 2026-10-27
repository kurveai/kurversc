"""Exact prediction alignment by an artifact's own masked entity/time keys."""

import hashlib

import numpy as np
import pandas as pd


def masked_query_keys(frame, *, entity, timestamp, target=None):
    if target is not None and target in frame:
        raise ValueError("Prediction provenance must not export hidden targets")
    values = frame[entity]
    if values.isna().any():
        raise ValueError("Unique complete masked entity/time keys required")
    if pd.api.types.is_integer_dtype(values.dtype):
        ids = values.to_numpy(dtype="uint64" if pd.api.types.is_unsigned_integer_dtype(values.dtype) else "int64")
    elif pd.api.types.is_float_dtype(values.dtype):
        ids = values.to_numpy(dtype="float64")
    else:
        ids = values.to_numpy()
        if ids.dtype == object:
            if not all(isinstance(v, str) for v in ids):
                raise ValueError("Explicit integral, floating or string entity identifiers required")
            ids = np.asarray(ids, dtype=str)
    keys = pd.DataFrame(dict(entity=ids,
        timestamp=pd.to_datetime(frame[timestamp], utc=True).dt.tz_localize(None).to_numpy()))
    query_positions(keys, keys)
    return keys


def query_positions(source, destination):
    for keys in (source, destination):
        if list(keys) != ["entity", "timestamp"] or keys.isna().any().any() or keys.duplicated().any():
            raise ValueError("Unique complete masked entity/time keys required")
    source_index, destination_index = pd.MultiIndex.from_frame(source), pd.MultiIndex.from_frame(destination)
    positions = source_index.get_indexer(destination_index)
    if len(source) != len(destination) or (positions < 0).any() or len(np.unique(positions)) != len(source):
        raise ValueError("Prediction artifact and query populations differ")
    return positions


def align_prediction(prediction, *, source_keys, query_keys):
    vector = np.asarray(prediction)
    if vector.shape != (len(source_keys),) or not np.isfinite(vector).all():
        raise ValueError("A finite vector aligned to its own source keys is required")
    return vector[query_positions(source_keys, query_keys)].copy()


def key_fingerprint(keys):
    query_positions(keys, keys)
    return hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).values.tobytes()).hexdigest()


def key_arrays(keys):
    query_positions(keys, keys)
    groups = keys.entity.to_numpy()
    if groups.dtype == object:
        if not all(isinstance(v, str) for v in groups):
            raise ValueError("Nonprimitive object identifiers cannot be saved as prediction keys")
        groups = np.asarray(groups, dtype=str)
    cutoffs = pd.to_datetime(keys.timestamp, utc=True).dt.tz_localize(None).to_numpy()
    return dict(groups=groups, forecast_cutoffs=cutoffs)


def artifact_keys(arrays):
    if not {"groups", "forecast_cutoffs"} <= set(arrays):
        raise ValueError("Unkeyed prediction artifact: cannot infer alignment from its score")
    values = np.asarray(arrays["groups"])
    if values.dtype == object:
        raise ValueError("Prediction key artifacts must not require pickle/object-array loading")
    keys = pd.DataFrame(dict(entity=values,
                             timestamp=pd.to_datetime(np.asarray(arrays["forecast_cutoffs"]))))
    query_positions(keys, keys)
    return keys
