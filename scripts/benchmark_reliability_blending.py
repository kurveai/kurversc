"""Train-only, entity-cross-fitted conditional challenger weights.

First-level out-of-fold predictions teach a small reliability head where an
optional readout helps. Coverage, support and disagreement are its only inputs;
entity keys route cross-fitting but are never predictors. Outer validation
labels are used only by the final paired three-origin/three-seed gate.
"""

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from kurversc.modeling import fit_final_catboost
from kurversc.temporal_policy import TemporalFitPolicy,fit_temporal_policy,_score
from scripts.benchmark_signal_challengers import incumbent_evidence,prepared,values
from scripts.benchmark_training_row_budget import stage_fold
from scripts.causal_parent_text_annotation import entity_folds
from scripts.engagement_local_comparison import context,protected,supplemental,freeze,TASK
from scripts.engagement_hypothesis_features import parent_text_columns
from scripts.gap_text_readout import FieldTextReadout
from scripts.fast_matched_bootstrap import accelerated_select_temporal_policy
from scripts.matched_learner_challengers import positioned_frame
from scripts.run_gap_followup_queue import dump,digest
from scripts.select_activity_readout import cache_locations


MENU_ALLOWANCE = 128


def reliability_features(frame,base,extra):
    """Ten bounded schema-generic covariates, never target or structural keys."""
    base,extra = np.asarray(base),np.asarray(extra)
    if base.shape != extra.shape or base.shape != (len(frame),) or not np.isfinite([base,extra]).all():
        raise ValueError("Aligned finite out-of-fold/forecast predictions required")
    if {"__event_entity","__event_cutoff"} & set(frame):
        raise ValueError("Structural routing keys must remain outside reliability predictors")
    numeric = frame.select_dtypes(include=["number"]).replace([np.inf,-np.inf],np.nan)
    coverage = np.zeros(len(frame))
    support = np.zeros(len(frame))
    if len(numeric.columns):
        coverage = numeric.notna().mean(axis=1).to_numpy()
        counts = [c for c in numeric if any(token in c.lower() for token in ("count","support","nunique"))]
        if counts:
            support = np.log1p(numeric[counts].clip(lower=0,upper=1e12).fillna(0)).mean(axis=1).to_numpy()
    text = [c for c in frame if pd.api.types.is_object_dtype(frame[c]) or isinstance(frame[c].dtype,pd.StringDtype)]
    text_coverage = np.zeros(len(frame))
    if text:
        text_coverage = np.mean([frame[c].astype("string").fillna("").replace("__missing__","").str.len().clip(upper=2048).to_numpy(float)/2048 for c in text],axis=0)
    return pd.DataFrame(dict(base=base,extra=extra,disagreement=np.abs(extra-base),
        signed_disagreement=extra-base,numeric_coverage=coverage,
        historical_support=support,text_coverage=text_coverage))


class ReliabilityWeight:
    def fit(self,covariates,truth,base,extra,*,task):
        if task not in ("classification","regression"):
            raise ValueError("Supported task required")
        truth,base,extra = map(np.asarray,(truth,base,extra))
        if len(covariates)!=len(truth) or not np.isfinite([truth,base,extra]).all():
            raise ValueError("Finite aligned mature OOF targets required")
        loss = (lambda p:(truth-p)**2) if task=="classification" else (lambda p:np.abs(truth-p))
        gain = loss(base)-loss(extra)
        self.scale = max(float(np.mean(np.abs(gain))),1e-8)
        target = np.clip(gain/self.scale,-5,5)
        positions = np.arange(len(target))
        if len(positions)>100_000:
            positions = np.sort(np.random.default_rng(0).choice(positions,100_000,replace=False))
        self.model = HistGradientBoostingRegressor(max_iter=100,max_depth=2,max_leaf_nodes=4,
            min_samples_leaf=max(200,len(positions)//200),l2_regularization=10,
            learning_rate=.05,random_state=0).fit(covariates.iloc[positions],target[positions])
        self.audit = dict(mature_oof_rows=len(truth),meta_fit_rows=len(positions),loss_scale=self.scale,
            task=task,features=list(covariates),maximum_depth=2,entity_key_is_predictor=False)
        return self

    def predict(self,covariates,maximum):
        if not 0 < maximum <= 1:
            raise ValueError("Interior bounded challenger allowance required")
        return maximum*np.clip(self.model.predict(covariates),0,1)


def group_masks(groups):
    codes = entity_folds(groups)
    masks = []
    for fold in range(3):
        fit,query = codes!=fold,codes==fold
        if not fit.any() or not query.any() or set(np.asarray(groups)[fit]) & set(np.asarray(groups)[query]):
            raise ValueError("Nonempty entity-disjoint OOF partitions required")
        masks.append((fit,query))
    return masks


def declared_root(metadata):
    """Canonical task declaration, not a workstation-specific cache path.

    Constructing a task accesses neither its hidden test table nor a scorer.
    Relocated research manifests intentionally keep their original provenance.
    """
    if metadata.get("entity_table"):
        return metadata["entity_table"]
    from relbench.tasks import get_task
    return get_task(metadata["dataset"],metadata["task"],download=False).entity_table


def fit_engagement_pair(ctx,folder):
    schema,target = ctx["metadata"]["feature_schema"],ctx["metadata"]["target"]
    train = positioned_frame(ctx["cache"]/"history.parquet",[*schema["columns"],target],ctx["positions"])
    held = pd.read_parquet(ctx["snapshots"]/f"fold{ctx['fold']}/validation.parquet",columns=schema["columns"])
    x = pd.concat([prepared(train,schema),supplemental(ctx,"training")],axis=1)
    vx = pd.concat([prepared(held,schema),supplemental(ctx,"validation")],axis=1)
    if list(x)!=ctx["order"]["columns"]:
        raise ValueError("Protected strong-head column order changed")
    root = declared_root(ctx["metadata"])
    columns = parent_text_columns(train,schema,root)
    if not columns:
        raise ValueError("No reusable schema-discovered parent text")
    y = ctx["train"][target].to_numpy()
    times = pd.to_datetime(ctx["train"].__event_cutoff)
    categories = [i for i,c in enumerate(x) if c in schema["categorical_columns"]]
    oof_base,oof_extra = np.empty(len(x)),np.empty(len(x))
    for group,(fit,query) in enumerate(group_masks(ctx["train"].__event_entity)):
        saved = folder/f"oof-group{group}.npz"
        if saved.exists():
            with np.load(saved) as a:
                np.testing.assert_array_equal(a["positions"],np.flatnonzero(query))
                np.testing.assert_array_equal(a["truth"],y[query])
                oof_base[query],oof_extra[query] = a["base"],a["extra"]
            continue
        text = FieldTextReadout(maximum_per_field=4096).fit(train.loc[fit],y[fit],columns)
        oof_extra[query] = text.predict_proba(train.loc[query])[:,1]
        members = []
        for n,cutoff in enumerate(ctx["cuts"]):
            mask = fit & times.eq(pd.Timestamp(cutoff)).to_numpy()
            model,_,_ = fit_final_catboost(x.loc[mask].reset_index(drop=True),pd.Series(y[mask]),
                task="classification",categorical=categories,random_state=0,
                model_params=ctx["manifest"]["model_params"])
            members.append(model.predict_proba(x.loc[query])[:,1])
            print(json.dumps(dict(phase="engagement_oof",fold=ctx["fold"],excluded_group=group,member=n,rows=int(mask.sum()))),flush=True)
            del model
            gc.collect()
        oof_base[query] = np.mean(members,axis=0)
        np.savez_compressed(saved,positions=np.flatnonzero(query),truth=y[query],base=oof_base[query],extra=oof_extra[query])
        del text,members
        gc.collect()
    text = FieldTextReadout(maximum_per_field=4096).fit(train,y,columns)
    text_prediction = text.predict_proba(held)[:,1]
    restored = pickle.loads(pickle.dumps(text))
    np.testing.assert_array_equal(text_prediction,restored.predict_proba(held)[:,1])
    meta = ReliabilityWeight().fit(reliability_features(x,oof_base,oof_extra),y,oof_base,oof_extra,task="classification")
    dump(folder/"meta-audit.json",dict(**meta.audit,columns=columns,
        latest_label_maturity=str((times+pd.Timedelta(ctx["metadata"]["label_horizon"])).max()),
        origin=ctx["phase"]["origin"],entity_crossfit_groups=3,changed_component="protected_strong_half_only"))
    del x,train,held,text,restored
    gc.collect()
    return meta,vx,text_prediction


def generic_inputs(args,fold):
    raw = json.loads((args.task_root/"raw-audit/manifest.json").read_text())
    metadata = raw["metadata"]
    policy = TemporalFitPolicy(**json.loads((args.task_root/"selection/results.json").read_text())["selected_policy"])
    cache,snapshots = cache_locations(args.task_root)
    raw["cache"],raw["phase_frozen_cache"] = str(cache),str(snapshots)
    columns = [metadata["target"],"__event_entity","__event_cutoff"]
    headers = pd.concat([pd.read_parquet(cache/n,columns=columns) for n in ("history.parquet","validation.parquet")],ignore_index=True)
    phase = stage_fold(args.output,raw,headers,fold,policy)
    folder = args.output/"inputs"/f"fold{fold}"
    train,held = [pd.read_parquet(folder/n) for n in ("train.parquet","validation.parquet")]
    return raw,metadata,policy,phase,train,held


def generic_fold(args,fold,folder):
    raw,metadata,policy,phase,train,held = generic_inputs(args,fold)
    kind = "classification" if metadata["metric"]=="roc_auc" else "regression"
    schema = metadata["feature_schema"]
    x,vx = prepared(train,schema),prepared(held,schema)
    y,times = train[metadata["target"]],pd.to_datetime(train.__event_cutoff)
    kwargs = dict(origin=phase["fit_origin"],label_horizon=metadata["label_horizon"],
        refit_frames=raw["observed_refit_frames"],task=kind,
        categorical=[i for i,c in enumerate(x) if c in schema["categorical_columns"]],
        model_params=raw["model_params"])
    candidate_policy = replace(policy,name="pooled_same_support",mode="pooled",blend_weight=1)
    budget = raw["row_limit"]
    if args.budget_validation:
        from scripts.confirm_selected_row_budget import locked_choice
        from scripts.benchmark_selected_row_budget import primitive_policy
        _,_,budget,_ = locked_choice(args.budget_validation)
        _,candidate_policy = primitive_policy(__import__("dataclasses").asdict(policy),raw["policies"])
    oof_base,oof_extra = np.empty(len(x)),np.empty(len(x))
    fixed = replace(policy,minimum_training_rows=1,training_frames=len(phase["initial_cutoffs"]))
    fixed_candidate = replace(candidate_policy,minimum_training_rows=1,training_frames=len(phase["initial_cutoffs"]))
    for group,(fit,query) in enumerate(group_masks(train.__event_entity)):
        path = folder/f"oof-group{group}.npz"
        if path.exists():
            with np.load(path) as a:
                np.testing.assert_array_equal(a["positions"],np.flatnonzero(query))
                np.testing.assert_array_equal(a["truth"],y.loc[query])
                oof_base[query],oof_extra[query] = a["base"],a["extra"]
            continue
        model,_ = fit_temporal_policy(x.loc[fit],y.loc[fit],times.loc[fit],policy=fixed,
            random_state=0,row_limit=raw["row_limit"],**kwargs)
        oof_base[query] = values(model,x.loc[query],kind)
        del model
        gc.collect()
        if args.budget_validation:
            primitive,_ = fit_temporal_policy(x.loc[fit],y.loc[fit],times.loc[fit],policy=fixed_candidate,
                random_state=0,row_limit=raw["row_limit"],**kwargs)
            old = values(primitive,x.loc[query],kind)
            del primitive
            gc.collect()
        model,_ = fit_temporal_policy(x.loc[fit],y.loc[fit],times.loc[fit],policy=fixed_candidate,
            random_state=0,row_limit=budget,**kwargs)
        extra = values(model,x.loc[query],kind)
        oof_extra[query] = oof_base[query]+policy.blend_weight*(extra-old) if args.budget_validation else extra
        np.savez_compressed(path,positions=np.flatnonzero(query),truth=y.loc[query],base=oof_base[query],extra=oof_extra[query])
        print(json.dumps(dict(phase="generic_oof",fold=fold,excluded_group=group,rows=int(fit.sum()))),flush=True)
        del model
        gc.collect()
    meta = ReliabilityWeight().fit(reliability_features(x,oof_base,oof_extra),y,oof_base,oof_extra,task=kind)
    dump(folder/"meta-audit.json",dict(**meta.audit,origin=phase["fit_origin"],
        latest_label_maturity=str((times+pd.Timedelta(metadata["label_horizon"])).max()),
        entity_crossfit_groups=3,original_initial_cutoffs=phase["initial_cutoffs"],
        original_refit_cutoffs=phase["observed_refit_cutoffs"]))
    cases = []
    for seed in range(3):
        control = incumbent_evidence(args.task_root/"raw-audit",fold,seed,policy)
        np.testing.assert_array_equal(control["truth"],held[metadata["target"]])
        np.testing.assert_array_equal(control["groups"],held.__event_entity)
        model,audit = fit_temporal_policy(x,y,times,policy=policy,random_state=seed,row_limit=raw["row_limit"],**kwargs)
        replay = values(model,vx,kind)
        np.testing.assert_allclose(replay,control["prediction"],atol=1e-9,rtol=0)
        del model
        gc.collect()
        if args.budget_validation:
            with np.load(args.budget_validation/f"fold{fold}/seed{seed}.npz") as a:
                np.testing.assert_array_equal(a["truth"],control["truth"])
                np.testing.assert_array_equal(a["groups"],control["groups"])
                extra = a[json.loads((args.budget_validation/"selection.json").read_text())["decision"]["selected"]].copy()
        else:
            model,_ = fit_temporal_policy(x,y,times,policy=candidate_policy,random_state=seed,row_limit=budget,**kwargs)
            extra = values(model,vx,kind)
            del model
            gc.collect()
        covariates = reliability_features(vx,replay,extra)
        predictions = {"conditional":replay+meta.predict(covariates,1)*(extra-replay),
                       "conditional_half":replay+meta.predict(covariates,.5)*(extra-replay),
                       "fixed_half":.5*(replay+extra)}
        if args.budget_validation:
            # The larger-row challenger already cleared validation. Reliability
            # must improve that accepted fixed recipe, not erase its gains while
            # merely beating the weaker 100K component. Never decide from test.
            control["prediction"] = extra.copy()
        cases.append(save_case(folder,fold,seed,raw["folds"][fold]["origin"],control,predictions,kind))
    return cases,kind


def save_case(folder,fold,seed,origin,control,predictions,kind):
    scores = {name:_score(control["truth"],pred,kind) for name,pred in {"control":control["prediction"],**predictions}.items()}
    np.savez_compressed(folder/f"seed{seed}.npz",truth=control["truth"],groups=control["groups"],
        control=control["prediction"],**predictions)
    r = dict(fold=fold,seed=seed,origin=origin,scores=scores,
        gains={n:(score-scores["control"])*(1 if kind=="classification" else -1) for n,score in scores.items() if n!="control"})
    dump(folder/f"seed{seed}.json",r)
    print(json.dumps(dict(phase="reliability_case",**r)),flush=True)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--engagement",action="store_true")
    p.add_argument("--budget-validation",type=Path)
    args = p.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    required = [args.task_root/"selection/results.json",args.task_root/"selection/manifest.json"]
    if args.budget_validation:
        required += [args.budget_validation/"selection.json",args.budget_validation/"selected-budget-identity.json"]
    freeze(args.output/"manifest.json",dict(task_root=str(args.task_root),engagement=args.engagement,
        budget_validation=str(args.budget_validation) if args.budget_validation else None,
        sources={p:digest(p) for p in [__file__,"scripts/engagement_local_comparison.py",
            "scripts/gap_text_readout.py","scripts/causal_parent_text_annotation.py","kurversc/temporal_policy.py"]},
        evidence_sources={str(p):digest(p) for p in required},
        meta_entity_folds=3,meta_seed=0,gate_origins=3,gate_seeds=3,menu_allowance=MENU_ALLOWANCE,
        scope="OOF mature training losses only; coverage/support/disagreement; no official/default changes"))
    records,kind = [],None
    for fold in (2,1,0):
        folder = args.output/f"fold{fold}"
        folder.mkdir(exist_ok=True)
        if all((folder/f"seed{s}.json").exists() for s in range(3)):
            records += [json.loads((folder/f"seed{s}.json").read_text()) for s in range(3)]
            kind = "classification" if args.engagement else ("classification" if json.loads((args.task_root/"raw-audit/manifest.json").read_text())["metadata"]["metric"]=="roc_auc" else "regression")
            continue
        if args.engagement:
            if args.task_root.resolve()!=TASK.resolve():
                raise ValueError("Protected engagement control identity differs")
            ctx = context(fold)
            meta,vx,text = fit_engagement_pair(ctx,folder)
            kind = "classification"
            for seed in range(3):
                strong,legacy = protected(ctx,seed)
                head = 2*strong-legacy
                covariates = reliability_features(vx,head,text)
                predictions = {"conditional":strong+.5*meta.predict(covariates,.5)*(text-head),
                    "conditional_half":strong+.5*meta.predict(covariates,.25)*(text-head),
                    "fixed_half":strong+.5*.25*(text-head)}
                records.append(save_case(folder,fold,seed,ctx["phase"]["origin"],
                    dict(truth=ctx["held"][ctx["metadata"]["target"]].to_numpy(),groups=ctx["held"].__event_entity.to_numpy(),prediction=strong),predictions,kind))
            del ctx,vx,meta,text
            gc.collect()
        else:
            cases,kind = generic_fold(args,fold,folder)
            records += cases
    recipes = [n for n in ("conditional","conditional_half","fixed_half") if all(r["gains"][n]>0 for r in records)]
    evidence = []
    for r in records:
        with np.load(args.output/f"fold{r['fold']}/seed{r['seed']}.npz") as a:
            for name,key in [("incumbent","control"),*[(n,n) for n in recipes]]:
                evidence.append(dict(origin=r["origin"],seed=r["seed"],policy=name,
                    truth=a["truth"].copy(),groups=a["groups"].copy(),prediction=a[key].copy()))
    decision = accelerated_select_temporal_policy(evidence,task=kind,minimum_origins=3,minimum_seeds=3,
        deployment_seed=0,bootstrap_repeats=1600,alpha=.05*len(recipes)/MENU_ALLOWANCE) if recipes else dict(selected="incumbent",reason="no_all_nine_positive_recipe")
    result = dict(decision=decision,records=records,eligible=recipes,complete_grid=len(records)==9,
        elapsed_seconds=time.monotonic()-started,default_changed=False,official_test_accessed=False,
        promotion="Validation evidence only; a separate independent refit/locked official confirmation is still required")
    dump(args.output/"results.json",result)
    print(json.dumps(dict(phase="reliability_complete",selected=decision["selected"])),flush=True)


if __name__ == "__main__":
    main()
