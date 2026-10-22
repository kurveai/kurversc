from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kurversc.temporal_policy import TemporalFitPolicy
from scripts import benchmark_signal_challengers as runner


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def setup_validation(tmp_path,monkeypatch,*,expanded):
    root = tmp_path / "task"
    output = tmp_path / "selection"
    frame = pd.DataFrame({"value":[.1,.9],"label":[0,1],"__event_cutoff":pd.to_datetime(["2020-01-01"]*2),
                          "__event_entity":[1,2]})
    (root / "cache").mkdir(parents=True)
    frame.to_parquet(root / "cache/history.parquet",index=False)
    frame.iloc[:0].to_parquet(root / "cache/validation.parquet",index=False)
    policy = TemporalFitPolicy("incumbent")
    metadata = dict(dataset="unregistered",task="arbitrary",target="label",metric="roc_auc",label_horizon="1d",
        feature_schema=dict(columns=["value"],categorical_columns=[],datetime_columns=[]),
        checkpoint_sha256="frozen",atomic_date_replay=True,fold="current",validation_cutoffs=1)
    save_json(root / "selection/results.json",dict(selected_policy=asdict(policy)))
    save_json(root / "selection/manifest.json",dict(metadata=metadata))
    folds = []
    for fold,origin in enumerate(("2020-01-10","2020-01-20")):
        folds.append(dict(origin=origin))
        folder = root / f"snapshots/fold{fold}"
        save_json(folder / "manifest.json",dict(origin=origin,snapshot_cutoff=origin,
            checkpoint_sha256="frozen",latest_observed_label_complete_at="2020-01-05"))
        frame.assign(__position=[0,1]).to_parquet(folder / "validation.parquet",index=False)
    save_json(root / "raw-audit/manifest.json",dict(folds=folds))
    calls = []
    monkeypatch.setattr(runner,"implementation_digest",lambda:"stable-head-implementation")
    monkeypatch.setattr(runner,"discover_exposure_path",lambda *a:None)
    monkeypatch.setattr(runner,"incumbent_evidence",lambda *a:dict(truth=np.array([0,1]),
        groups=np.array([1,2]),prediction=np.array([.2,.8])))
    monkeypatch.setattr(runner,"select_temporal_policy",lambda *a,**k:dict(selected="incumbent",candidates={}))
    def fit(spec,*a,**kw):
        calls.append((spec.kind,spec.training_frames,spec.regularized,kw["origin"],kw["seed"]))
        return np.array([.4,.6]),dict(raw_head=True)
    monkeypatch.setattr(runner,"fit_candidate",fit)
    import kurversc.task_source
    source = SimpleNamespace(task=SimpleNamespace(entity_table="people",timedelta=pd.Timedelta("1d")),
        inner_split=lambda:SimpleNamespace(db_state=SimpleNamespace(table_dict={}),cutoff=pd.Timestamp("2020-01-20")))
    monkeypatch.setattr(kurversc.task_source,"load_task_source",lambda *a,**k:source)
    return SimpleNamespace(task_root=root,output=output,threads=4,exposure_blends=expanded),calls


def test_validation_applies_partial_exposure_weights_and_reuses_raw_heads(tmp_path,monkeypatch):
    args,calls = setup_validation(tmp_path,monkeypatch,expanded=True)
    result,_ = runner.validate(args)
    assert len(calls)==len(set(calls))==7*2*3
    assert len(result["manifest"]["menu"])==12
    for weight in (.125,.5):
        with np.load(args.output / f"fold1-exposure_quality__blend_{weight:g}-seed0.npz") as a:
            np.testing.assert_array_equal(a["prediction"],(1-weight)*np.array([.2,.8])+weight*np.array([.4,.6]))
    calls.clear()
    runner.validate(args)
    assert not calls


def test_disabled_option_preserves_legacy_candidate_vectors_and_menu(tmp_path,monkeypatch):
    args,calls = setup_validation(tmp_path,monkeypatch,expanded=False)
    result,_ = runner.validate(args)
    assert len(result["manifest"]["menu"])==8
    with np.load(args.output / "fold1-exposure_quality-seed0.npz") as a:
        np.testing.assert_array_equal(a["prediction"],[.4,.6])
    with np.load(args.output / "fold1-text25-seed0.npz") as a:
        np.testing.assert_array_equal(a["prediction"],.75*np.array([.2,.8])+.25*np.array([.4,.6]))


def test_inference_honors_the_frozen_menu_weight_once_after_key_alignment(tmp_path,monkeypatch):
    from kurversc.prediction_provenance import masked_query_keys,key_arrays,key_fingerprint
    from kurversc.signal_challengers import SignalChallenger

    args,_ = setup_validation(tmp_path,monkeypatch,expanded=False)
    args.output.mkdir(parents=True)
    origin = pd.Timestamp("2020-02-01")
    original = pd.DataFrame({"who":[1,2],"when":[origin,origin]})
    keys = masked_query_keys(original,entity="who",timestamp="when")
    baseline = args.task_root / "official"
    baseline.mkdir()
    save_json(baseline / "results.json",dict(official_test_scores=dict(candidate=.5),arm="incumbent",
        query_keys_sha256=key_fingerprint(keys)))
    np.savez_compressed(baseline / "predictions.npz",candidate=[.2,.8],**key_arrays(keys))
    spec = SignalChallenger("frozen-custom-half-weight-not-in-current-menu","exposure",weight=.5)
    selected = dict(selected=spec.name,manifest=dict(base_policy=asdict(TemporalFitPolicy("incumbent")),
        menu=[asdict(spec)]),decision=dict(candidates={spec.name:dict(reason="eligible")}))
    save_json(args.output / "selection.json",selected)
    save_json(args.output / "applicability.json",dict(exposure_path=dict(parent="sessions",parent_root_fk="owner",
        event="events",event_parent_fk="session",object_table="objects",object_fk="object",response="response")))
    train = pd.read_parquet(args.task_root / "cache/history.parquet")
    forecast = pd.DataFrame({"__event_entity":[2,1],"__event_cutoff":[origin,origin],"value":[.1,.9]})
    expected = np.array([.7,.3])
    calls = []
    def evaluate(prediction,*,target_table=None):
        assert target_table is None
        with np.load(args.output / "predictions.npz") as saved:
            np.testing.assert_array_equal(saved["candidate"],prediction)
            np.testing.assert_allclose(saved["candidate"],expected)
        calls.append(prediction)
        return dict(roc_auc=.75)
    task = SimpleNamespace(entity_table="people",entity_col="who",time_col="when",timedelta=pd.Timedelta("1d"),evaluate=evaluate)
    source = SimpleNamespace(task=task)
    outer = SimpleNamespace(cutoff=origin,db_state=SimpleNamespace(table_dict={}),eval_table=SimpleNamespace(df=original))
    monkeypatch.setattr(runner,"materialize_outer",lambda *a:(source,outer,train,forecast))
    monkeypatch.setattr(runner,"fit_candidate",lambda *a,**kw:(np.array([.6,.4]),dict(raw_head=True)))
    class Provider:
        last_audit = {}
        def __init__(self,*a,**kw):
            pass
        def transform(self,frame,**kw):
            return pd.DataFrame({"extra":np.zeros(len(frame))})
        def close(self):
            pass
    monkeypatch.setattr(runner,"SchemaExposureQuality",Provider)
    metadata = dict(dataset="unregistered",task="arbitrary",target="label",metric="roc_auc",
        feature_schema=dict(columns=["value"],categorical_columns=[],datetime_columns=[]))
    result = runner.official(args,selected,metadata)
    assert result["arm"]==spec.name and len(calls)==1
