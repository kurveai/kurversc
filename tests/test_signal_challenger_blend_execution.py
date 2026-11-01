"""Accepted heads retain identical weighting in validation and inference."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kurversc.signal_challengers import ChallengerUnavailable
from kurversc.temporal_policy import TemporalFitPolicy
from scripts import benchmark_generic_readouts as runner


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_validation(tmp_path, monkeypatch, *, payload_available=True):
    root, output = tmp_path / "task", tmp_path / "selection"
    frame = pd.DataFrame({"value": [.1, .9], "label": [0, 1],
        "__event_cutoff": pd.to_datetime(["2020-01-01"] * 2), "__event_entity": [1, 2]})
    policy = TemporalFitPolicy("incumbent")
    metadata = dict(dataset="unregistered", task="arbitrary", entity_table="people",
        target="label", metric="roc_auc", label_horizon="1d", validation_cutoffs=1,
        feature_schema=dict(columns=["value"], categorical_columns=[], datetime_columns=[]))
    save_json(root / "selection/results.json", dict(selected_policy=asdict(policy)))
    save_json(root / "selection/manifest.json", dict(metadata=metadata))
    folds = [dict(origin=origin) for origin in ("2020-01-10", "2020-01-20")]
    save_json(root / "raw-audit/manifest.json", dict(folds=folds))

    calls = []
    monkeypatch.setattr(runner, "resolve_task_metadata", lambda meta: meta)
    monkeypatch.setattr(runner, "incumbent_evidence", lambda *a: dict(
        truth=np.array([0, 1]), groups=np.array([1, 2]), prediction=np.array([.2, .8])))
    monkeypatch.setattr(runner, "select_temporal_policy",
        lambda *a, **k: dict(selected="incumbent", candidates={}))
    monkeypatch.setattr(runner, "phase_source_root", lambda *a, **k: tmp_path / "source")

    def phase_frames(task_root, raw, selected, fold, **kwargs):
        return frame, frame, dict(origin=folds[fold]["origin"],
            fit_origin=folds[fold]["origin"], optional_fit_available=payload_available)

    def fit(spec, *a, **kw):
        calls.append((spec.kind, spec.training_frames, kw["origin"], kw["seed"]))
        return np.array([.4, .6]), dict(raw_head=True)

    def no_objects(*a, **kw):
        raise ChallengerUnavailable("No declared object path")

    monkeypatch.setattr(runner, "phase_frames", phase_frames)
    monkeypatch.setattr(runner, "fit_signal", fit)
    monkeypatch.setattr(runner, "object_payload", no_objects)
    return SimpleNamespace(task_root=root, output=output, threads=4), calls, frame


def test_validation_blends_text_once_and_reuses_shared_raw_heads(tmp_path, monkeypatch):
    args, calls, _ = setup_validation(tmp_path, monkeypatch)
    result, _ = runner.validate(args)
    # One text head serves two weights; three/five-frame heads remain distinct.
    assert len(calls) == len(set(calls)) == 3 * 2 * 3
    assert len(result["records"]) == 6
    with np.load(args.output / "fold1/seed0.npz") as saved:
        for name, weight in (("text25", .25), ("text50", .5)):
            np.testing.assert_array_equal(saved[name],
                (1 - weight) * np.array([.2, .8]) + weight * np.array([.4, .6]))
        np.testing.assert_array_equal(saved["frame_ensemble3"], [.4, .6])
        assert not any("exposure" in name for name in saved.files)
    calls.clear()
    runner.validate(args)
    assert not calls


def test_unavailable_payload_keeps_the_incumbent_and_records_why(tmp_path, monkeypatch):
    args, calls, _ = setup_validation(tmp_path, monkeypatch, payload_available=False)
    result, _ = runner.validate(args)
    assert result["selected"] == "incumbent" and not calls
    for record in result["records"]:
        assert record["unavailable"] == {"payload_learners": "global_optional_feature_cell_capacity"}
    with np.load(args.output / "fold1/seed0.npz") as saved:
        assert set(saved.files) == {"truth", "groups", "prediction"}
        np.testing.assert_array_equal(saved["prediction"], [.2, .8])


def test_inference_blends_admitted_text_once_after_key_alignment(tmp_path, monkeypatch):
    from kurversc.prediction_provenance import masked_query_keys, key_arrays, key_fingerprint

    args, _, train = setup_validation(tmp_path, monkeypatch)
    args.output.mkdir(parents=True)
    origin = pd.Timestamp("2020-02-01")
    original = pd.DataFrame({"who": [1, 2], "when": [origin, origin]})
    keys = masked_query_keys(original, entity="who", timestamp="when")
    baseline = args.task_root / "official"
    baseline.mkdir()
    save_json(baseline / "results.json", dict(official_test_scores=dict(candidate=.5),
        query_keys_sha256=key_fingerprint(keys)))
    np.savez_compressed(baseline / "predictions.npz", candidate=[.2, .8], **key_arrays(keys))
    selected = dict(selected="text50", policy=asdict(TemporalFitPolicy("incumbent")))
    save_json(args.output / "selection.json", selected)
    forecast = pd.DataFrame({"__event_entity": [2, 1],
        "__event_cutoff": [origin, origin], "value": [.1, .9]})
    calls = []

    def evaluate(prediction, *, target_table=None):
        assert target_table is None
        with np.load(args.output / "predictions.npz") as saved:
            np.testing.assert_array_equal(saved["candidate"], prediction)
            np.testing.assert_allclose(saved["candidate"], [.7, .3])
        calls.append(prediction)
        return dict(roc_auc=.75)

    task = SimpleNamespace(entity_table="people", entity_col="who", time_col="when",
        target_col="label", evaluate=evaluate)
    source = SimpleNamespace(task=task)
    outer = SimpleNamespace(cutoff=origin, eval_table=SimpleNamespace(df=original))
    monkeypatch.setattr(runner, "materialize_outer", lambda *a: (source, outer, train, forecast))
    monkeypatch.setattr(runner, "fit_signal", lambda *a, **kw: (np.array([.6, .4]), dict(raw_head=True)))
    metadata = json.loads((args.task_root / "selection/manifest.json").read_text())["metadata"]
    result = runner.official(args, selected, metadata)
    assert result["arm"] == "text50" and len(calls) == 1
