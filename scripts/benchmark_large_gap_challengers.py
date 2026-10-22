"""Frozen-graph large-gap hypotheses with exact fallback and strict admission.

No official/test split is opened. Target outcomes are used only for mature
training and retrospective scoring. New features are target-free. A latest
seed-zero scout can reject early, never promote early. Every admitted recipe
needs every frozen origin x seeds 0/1/2 and paired entity-cluster confidence.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from kurversc import temporal_policy
from kurversc.temporal_policy import TemporalFitPolicy, fit_temporal_policy, _score
from scripts.activity_timeline_challenger import table_metadata
from scripts.benchmark_phase_frozen_breadth_v2 import task_root_table
from scripts.benchmark_signal_challengers import incumbent_evidence, prepared, values
from scripts.benchmark_training_row_budget import stage_fold
from scripts.engagement_local_comparison import freeze
from scripts.frozen_auc_bootstrap import frozen_auc_select_temporal_policy
from scripts.marked_response_history import MarkedResponseHistory, ResponsePathUnavailable
from scripts.run_gap_followup_queue import digest, dump
from scripts.select_activity_readout import cache_locations
from scripts.zero_aware_quantile_readout import ZeroAwareQuantileReadout


WEIGHTS = (.125,.5,1.)
ARMS = ("zero_aware","response_timing","response_context","response_combined")
MENU_ALLOWANCE = 200
SOURCES = [__file__,"scripts/marked_response_history.py","scripts/zero_aware_quantile_readout.py",
    "scripts/activity_timeline_challenger.py","scripts/benchmark_phase_frozen_breadth_v2.py",
    "scripts/benchmark_signal_challengers.py","scripts/benchmark_training_row_budget.py",
    "scripts/frozen_auc_bootstrap.py","scripts/fast_matched_bootstrap.py",
    "kurversc/temporal_policy.py","kurversc/modeling.py"]


def protected_blend(control,candidate,support,weight):
    control,candidate=np.asarray(control),np.asarray(candidate)
    support=np.asarray(support,dtype=bool)
    if (control.ndim!=1 or candidate.shape!=control.shape or support.shape!=control.shape
            or not np.isfinite(candidate).all() or not np.isfinite(control).all() or weight not in WEIGHTS):
        raise ValueError("Aligned finite protected predictions and a declared blend required")
    result=control.copy()
    result[support]=(1-weight)*control[support]+weight*candidate[support]
    np.testing.assert_array_equal(result[~support],control[~support])
    return result


def complete_positive_recipes(records,recipes,folds):
    expected={(f,s) for f in folds for s in range(3)}
    keys=[(r["fold"],r["seed"]) for r in records]
    if len(keys)!=len(set(keys)):
        raise ValueError("Duplicate research cases")
    if set(keys)!=expected:
        return []
    return [name for name in recipes if all(r["gains"][name]>0 for r in records)]


@contextmanager
def isolated_zero_fitter():
    """Substitute only this worker's estimator factory, restored on every exit."""
    original=temporal_policy.fit_final_catboost

    def fit(features,labels,*,task,categorical,random_state=0,model_params=None,**kwargs):
        if task!="regression":
            raise ValueError("Zero-aware research factory requires numeric regression")
        model=ZeroAwareQuantileReadout(seed=random_state,model_params=model_params).fit(
            features,labels,categorical=categorical)
        return model,None,None

    temporal_policy.fit_final_catboost=fit
    try:
        yield
    finally:
        temporal_policy.fit_final_catboost=original


def fit_prediction(x,y,times,vx,*,raw,meta,policy,phase,seed,zero=False):
    kwargs=dict(origin=phase["fit_origin"],label_horizon=meta["label_horizon"],
        refit_frames=raw["observed_refit_frames"],policy=policy,
        task="classification" if meta["metric"]=="roc_auc" else "regression",
        categorical=[i for i,c in enumerate(x) if c in meta["feature_schema"]["categorical_columns"]],
        random_state=seed,row_limit=raw["row_limit"],model_params=raw["model_params"])
    if zero:
        kwargs["policy"]=replace(policy,name="zero_aware_same_support",readout="direct",blend_weight=1.)
        with isolated_zero_fitter():
            model,audit=fit_temporal_policy(x,y,times,**kwargs)
    else:
        model,audit=fit_temporal_policy(x,y,times,**kwargs)
    prediction=values(model,vx,kwargs["task"])
    np.testing.assert_array_equal(prediction,values(pickle.loads(pickle.dumps(model)),vx,kwargs["task"]))
    del model
    gc.collect()
    return prediction,audit


def source_identity(paths):
    return {str(p):digest(p) for p in paths}


def frame_identity(frame,columns):
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns],index=False).values.tobytes()).hexdigest()


def run(args):
    started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    raw=json.loads((args.task_root/"raw-audit/manifest.json").read_text())
    meta=raw["metadata"]
    if meta["fold"]!="current" or not meta["atomic_date_replay"]:
        raise ValueError("Corrected point-in-time frozen graph required")
    if meta!=json.loads((args.task_root/"selection/manifest.json").read_text())["metadata"]:
        raise ValueError("Selected graph and raw audit differ")
    cache,snapshots=cache_locations(args.task_root)
    raw={**raw,"cache":str(cache),"phase_frozen_cache":str(snapshots)}
    policy=TemporalFitPolicy(**json.loads((args.task_root/"selection/results.json").read_text())["selected_policy"])
    folds=list(range(len(raw["folds"])))
    if len(folds)<2:
        raise ValueError("At least two available frozen origins required")
    arms=args.arms.split(",")
    if len(set(arms))!=len(arms) or set(arms)-set(ARMS):
        raise ValueError("Declared unique research arm menu required")
    recipes=[f"{arm}_{weight:g}" for arm in arms for weight in WEIGHTS]
    manifest=dict(metadata=meta,policy=asdict(policy),folds=folds,seeds=[0,1,2],
        model_params=raw["model_params"],row_limit=raw["row_limit"],arms=arms,
        recipes=recipes,menu_allowance=MENU_ALLOWANCE,
        sources=source_identity(SOURCES),input_files=source_identity([
            args.task_root/"raw-audit/manifest.json",args.task_root/"selection/results.json",
            cache/"history.parquet",cache/"validation.parquet",
            *[snapshots/f"fold{f}/manifest.json" for f in folds],
            *[snapshots/f"fold{f}/validation.parquet" for f in folds]]),
        default_changed=False,official_test_accessed=False,
        zero_aware_compute="Two 300-tree heads per member; not an equal-compute claim",
        scope="Isolated validation-only hypothesis panel; exact unsupported/failed incumbent fallback; no guarantee under future shift")
    freeze(args.output/"manifest.json",manifest)
    target=meta["target"]
    headers=pd.concat([pd.read_parquet(cache/name,columns=[target,"__event_entity","__event_cutoff"])
        for name in ("history.parquet","validation.parquet")],ignore_index=True)
    task_kind="classification" if meta["metric"]=="roc_auc" else "regression"
    root=None
    if any(arm.startswith("response_") for arm in arms):
        source_table=table_metadata(args.source_root/"tasks"/meta["task"]/"train.parquet")
        root=task_root_table(meta,source_table)
    records,evidence,applicability=[],[],[]
    active_recipes=set(recipes)
    # Newest origin first. A negative scout suppresses further fits for that
    # preset but cannot create a successful decision from partial evidence.
    for fold in reversed(folds):
        folder=args.output/f"fold{fold}"
        folder.mkdir(exist_ok=True)
        if not active_recipes:
            break
        stage_fold(folder,raw,headers,fold,policy)
        staged=folder/"inputs"/f"fold{fold}"
        train,held=[pd.read_parquet(staged/name) for name in ("train.parquet","validation.parquet")]
        phase=json.loads((staged/"phase.json").read_text())
        if (pd.to_datetime(train.__event_cutoff)+pd.Timedelta(meta["label_horizon"])>pd.Timestamp(phase["fit_origin"])).any():
            raise ValueError("Immature training outcomes reached a frozen phase")
        x,vx=prepared(train,meta["feature_schema"]),prepared(held,meta["feature_schema"])
        y,times=train[target],pd.to_datetime(train.__event_cutoff)
        extras,query_extras,supports={}, {}, {}
        fold_active=[arm for arm in arms if any(f"{arm}_{w:g}" in active_recipes for w in WEIGHTS)]
        numeric=pd.to_numeric(y).to_numpy(float)
        zero_applicable=(task_kind=="regression" and np.isfinite(numeric).all() and (numeric>=0).all()
                         and np.sum(numeric==0)>=20 and np.sum(numeric>0)>=20)
        if "zero_aware" in fold_active and not zero_applicable:
            applicability.append(dict(fold=fold,arm="zero_aware",eligible=False,reason="unsupported_target_domain"))
            active_recipes-={f"zero_aware_{w:g}" for w in WEIGHTS}
            fold_active.remove("zero_aware")
        marked=[arm for arm in fold_active if arm.startswith("response_")]
        if marked:
            print(json.dumps(dict(phase="marked_response_features",fold=fold,rows=len(train),root=root)),flush=True)
            try:
                provider=MarkedResponseHistory(args.source_root/"db",root,
                    training_cutoff=times.min(),horizon=meta["label_horizon"],temporary=folder/"spill",threads=2)
            except ResponsePathUnavailable as error:
                for arm in marked:
                    active_recipes-={f"{arm}_{w:g}" for w in WEIGHTS}
                    fold_active.remove(arm)
                    applicability.append(dict(fold=fold,arm=arm,eligible=False,reason=str(error)))
            else:
                try:
                    extra=provider.transform(train,phase_boundary=phase["fit_origin"])
                    training_audit=provider.last_audit
                    query_extra=provider.transform(held,phase_boundary=phase["fit_origin"])
                    query_audit=provider.last_audit
                    query_support={k:v.copy() for k,v in provider.support.items()}
                finally:
                    provider.close()
                extra.to_parquet(folder/"response-training.parquet",index=False)
                query_extra.to_parquet(folder/"response-validation.parquet",index=False)
                initial=times.isin(pd.to_datetime(phase["initial_cutoffs"]))
                selected=[c for c in extra if extra.loc[initial.to_numpy(),c].nunique(dropna=False)>1]
                for arm in marked:
                    columns=[c for c in selected if arm=="response_combined" or
                        (c.startswith("mr_") if arm=="response_timing" else c.startswith("mc_"))]
                    if not columns:
                        active_recipes-={f"{arm}_{w:g}" for w in WEIGHTS}
                        fold_active.remove(arm)
                        applicability.append(dict(fold=fold,arm=arm,eligible=False,reason="no_initial_variable_payload"))
                        continue
                    extras[arm],query_extras[arm]=extra[columns],query_extra[columns]
                    supports[arm]=query_support[arm.split("response_")[1]]
                dump(folder/"response-audit.json",dict(training=training_audit,validation=query_audit,
                    payload={k:list(v.columns) for k,v in extras.items()},
                    training_row_hash=frame_identity(train,[target,"__event_entity","__event_cutoff"]),
                    validation_row_hash=frame_identity(held,[target,"__event_entity","__event_cutoff"])))
        if not fold_active:
            continue
        for seed in range(3):
            if not active_recipes:
                break
            archived=incumbent_evidence(args.task_root/"raw-audit",fold,seed,policy)
            np.testing.assert_array_equal(archived["truth"],held[target])
            np.testing.assert_array_equal(archived["groups"],held.__event_entity)
            print(json.dumps(dict(phase="exact_control_replay",fold=fold,seed=seed,rows=len(train))),flush=True)
            native,audit=fit_prediction(x,y,times,vx,raw=raw,meta=meta,policy=policy,phase=phase,seed=seed)
            np.testing.assert_allclose(native,archived["prediction"],rtol=0,atol=1e-9)
            arrays=dict(truth=archived["truth"],groups=archived["groups"],control=archived["prediction"],
                        forecast_cutoffs=pd.to_datetime(held.__event_cutoff).to_numpy())
            evidence.append(dict(origin=phase["origin"],seed=seed,policy="incumbent",**archived))
            for arm in arms:
                possible=[f"{arm}_{w:g}" for w in WEIGHTS if f"{arm}_{w:g}" in active_recipes]
                if arm not in fold_active or not possible:
                    for weight in WEIGHTS:
                        arrays[f"{arm}_{weight:g}"]=arrays["control"].copy()
                    continue
                print(json.dumps(dict(phase="challenger_fit",fold=fold,seed=seed,arm=arm)),flush=True)
                try:
                    if arm=="zero_aware":
                        pred,_=fit_prediction(x,y,times,vx,raw=raw,meta=meta,policy=policy,phase=phase,seed=seed,zero=True)
                        support=np.ones(len(held),bool)
                    else:
                        xx=pd.concat([x.reset_index(drop=True),extras[arm]],axis=1)
                        vxx=pd.concat([vx.reset_index(drop=True),query_extras[arm]],axis=1)
                        pred,_=fit_prediction(xx,y,times,vxx,raw=raw,meta=meta,policy=policy,phase=phase,seed=seed)
                        support=supports[arm]
                except ValueError as error:
                    if arm!="zero_aware" or "positive and zero" not in str(error):
                        raise
                    pred,support=arrays["control"],np.zeros(len(held),bool)
                    applicability.append(dict(fold=fold,seed=seed,arm=arm,eligible=False,reason=str(error)))
                for weight in WEIGHTS:
                    name=f"{arm}_{weight:g}"
                    candidate=protected_blend(arrays["control"],pred,support,weight)
                    arrays[name]=candidate
                    evidence.append(dict(origin=phase["origin"],seed=seed,policy=name,
                        truth=arrays["truth"],groups=arrays["groups"],prediction=candidate))
            scores={name:_score(arrays["truth"],pred,task_kind) for name,pred in arrays.items()
                    if name not in {"truth","groups","forecast_cutoffs"}}
            direction=1 if task_kind=="classification" else -1
            gains={name:direction*(score-scores["control"]) for name,score in scores.items() if name!="control"}
            report=dict(fold=fold,seed=seed,origin=phase["origin"],scores=scores,gains=gains,
                exact_control_max_error=float(np.max(np.abs(native-arrays["control"]))),
                control_audit=audit,manifest_sha256=digest(args.output/"manifest.json"))
            np.savez_compressed(folder/f"seed{seed}.npz",**arrays)
            dump(folder/f"seed{seed}.json",report)
            records.append(report)
            dump(args.output/"progress.json",dict(records=records,applicability=applicability,active_recipes=sorted(active_recipes)))
            print(json.dumps({k:v for k,v in report.items() if k!="control_audit"}),flush=True)
            # Nonpositive paired evidence rejects a preset without spending
            # further compute. Rejected vectors remain saved and reported.
            active_recipes={name for name in active_recipes if gains.get(name,0)>0}
        del train,held,x,vx,extras,query_extras
        gc.collect()
    complete=complete_positive_recipes(records,recipes,folds)
    admitted={"selected":"incumbent","reason":"incomplete_or_nonpositive_grid","candidates":{}}
    if complete:
        included=[r for r in evidence if r["policy"]=="incumbent" or r["policy"] in complete]
        admitted=frozen_auc_select_temporal_policy(included,task=task_kind,minimum_origins=len(folds),
            minimum_seeds=3,deployment_seed=0,bootstrap_repeats=1600,
            alpha=.05*max(1,len(complete))/MENU_ALLOWANCE)
        selected=admitted["selected"]
        if selected!="incumbent":
            recent=admitted["candidates"][selected]["folds"][-1]
            if recent.get("recent_lower_bound",0)<=0:
                admitted={**admitted,"selected":"incumbent","reason":"no_confident_recent_gain"}
    result=dict(records=records,complete_positive_recipes=complete,decision=admitted,
        applicability=applicability,folds=folds,seeds=[0,1,2],elapsed_seconds=time.monotonic()-started,
        manifest_sha256=digest(args.output/"manifest.json"),official_test_accessed=False,default_changed=False,
        promotion_scope="A task-level research gate only; broader regression checks and locked official replay still required")
    if source_identity(SOURCES)!=manifest["sources"]:
        raise ValueError("Frozen implementation changed during experiment")
    dump(args.output/"results.json",result)
    print(json.dumps(dict(phase="complete",selected=admitted["selected"],reason=admitted["reason"],
        complete_positive_recipes=complete,elapsed_seconds=result["elapsed_seconds"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("task-root","source-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--arms",default=",".join(ARMS))
    run(parser.parse_args())
