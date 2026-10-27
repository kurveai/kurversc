"""Persist the actual sampled estimator inputs without retaining feature frames."""

from contextlib import contextmanager
import copy

import numpy as np
import pandas as pd

import kurversc.core as core
from scripts.probe_arrow_frozen_replay import file_hash
from scripts.run_gap_followup_queue import dump


@contextmanager
def capture_sampled_inputs(output):
    original = core._fit_temporal_frame_estimator
    records, by_model = [], {}

    def fit(features, labels, **kwargs):
        if kwargs["model_backend"] != "catboost" or kwargs["task"] != "regression":
            raise ValueError("Frozen count replay expects ordinary CatBoost regression")
        number = len(records)
        folder = output / f"member{number}"
        folder.mkdir(parents=True, exist_ok=True)
        # features already passed through the ORIGINAL core row sampler. Do
        # not sample or join again, and never assume these are full root frames.
        features.to_parquet(folder / "features.parquet", index=False)
        pd.DataFrame(dict(target=np.asarray(labels))).to_parquet(folder / "labels.parquet", index=False)
        model, seconds, classes = original(features, labels, **kwargs)
        record = dict(member=number, rows=len(features), columns=list(features),
            categorical=list(kwargs["categorical"]), seed=int(kwargs["random_state"]),
            captured_driver_params=copy.deepcopy(kwargs["model_params"]),
            source="actual original core estimator call, after sampling",
            features_sha256=file_hash(folder / "features.parquet"),
            labels_sha256=file_hash(folder / "labels.parquet"))
        records.append(record)
        by_model[id(model)] = number
        dump(folder / "audit.json", record)
        dump(output / "sampled-members.json", records)
        return model, seconds, classes

    core._fit_temporal_frame_estimator = fit
    try:
        yield records, by_model
    finally:
        core._fit_temporal_frame_estimator = original


@contextmanager
def capture_masked_frames(output, schema, *, entity, root):
    original = core._materialize_at_cutoff
    records = []

    def materialize(*args, **kwargs):
        frame = original(*args, **kwargs)
        orders = [n for n in frame if n.startswith("__kurversc_prediction_order__")]
        key = entity if entity in frame else f"{root}_{entity}"
        if len(orders) != 1 or key not in frame:
            raise ValueError("Unique original masked-query order and entity key required")
        saved = pd.DataFrame({n: frame[n] for n in schema["columns"]}, copy=False)
        saved["__event_entity"] = frame[key].to_numpy()
        saved["__event_cutoff"] = str(pd.Timestamp(kwargs["cut_date"]))
        saved["__query_order"] = frame[orders[0]].to_numpy()
        path = output / f"forecast{len(records)}.parquet"
        saved.to_parquet(path, index=False)
        records.append(dict(path=str(path), rows=len(saved), sha256=file_hash(path)))
        dump(output / "masked-frames.json", records)
        return frame

    core._materialize_at_cutoff = materialize
    try:
        yield records
    finally:
        core._materialize_at_cutoff = original


def selected_members(estimator, by_model):
    from kurversc.modeling import FrameEnsemble

    if not isinstance(estimator, FrameEnsemble):
        raise ValueError("The original final estimator must remain a frame ensemble")
    members = [by_model.get(id(model)) for model in estimator.models]
    if not members or None in members or len(set(members)) != len(members):
        raise ValueError("Final models must map one-to-one onto their actual sampled calls")
    return members


def canonical_vectors(forecasts, vectors, *, expected_rows):
    order = np.concatenate([f.__query_order.to_numpy() for f in forecasts])
    if sorted(order.tolist()) != list(range(expected_rows)):
        raise ValueError("Masked queries must appear exactly once, in original row order")
    positions = np.argsort(order, kind="stable")
    keys = pd.concat([f[["__event_entity", "__event_cutoff"]] for f in forecasts], ignore_index=True).iloc[positions].reset_index(drop=True)
    output = {}
    for name, chunks in vectors.items():
        vector = np.concatenate(chunks)
        if vector.shape != (expected_rows,) or not np.isfinite(vector).all():
            raise ValueError("Finite full-cohort predictions required")
        output[name] = vector[positions]
    return keys, output
