"""One validation-selected menu of generic post-search prediction candidates.

Every candidate challenges the same temporal-policy incumbent on the same
origins and seeds. Applicability depends on inputs, never earlier admissions.
The candidate menu and one selector are shared across schemas and task types.
"""

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from kurversc.prediction_provenance import (
    align_prediction, artifact_keys, key_arrays, key_fingerprint, masked_query_keys,
)
from kurversc.signal_challengers import (
    BoundedTextReadout, ChallengerUnavailable, SignalChallenger,
    apply_challenger_weight, challenger_menu, challenger_policy, mature_support_mask,
)
from kurversc.task_source import CONTEXT_ENV, ContextTaskSource, phase_source_root
from kurversc.temporal_policy import (
    InsufficientPolicySupport, TemporalFitPolicy, _score, fit_temporal_policy,
    select_temporal_policy,
)
from scripts.discrete_target_projection import TargetDomainProjection
from scripts.generic_staged_helpers import (
    MAX_OPTIONAL_FEATURE_CELLS, digest, dump, fit_prediction, freeze,
    incumbent_evidence, materialize_outer, object_payload, phase_frames,
    prepared, values,
)


WEIGHTS = (.125, .5, 1.)
SIGNALS = challenger_menu()
RECIPES = tuple(s.name for s in SIGNALS) + (
    "nonnegative", "integer_counts", "object_full", "object_half",
    *(f"zero_aware_{w:g}" for w in WEIGHTS),
)
CONTRACT = dict(
    candidates=RECIPES, seeds=[0, 1, 2], deployment_seed=0,
    selector="select_temporal_policy", minimum_origins=2, minimum_seeds=3,
    alpha=.05, bootstrap_repeats=600,
    classification_loss_cap=.005, regression_relative_loss_cap=.01,
    maximum_feature_cells=MAX_OPTIONAL_FEATURE_CELLS,
    eligibility_depends_on_previous_admissions=False,
    benchmark_score_dependent_choices=False,
)


def resolve_task_metadata(metadata):
    """Bind the root role to the declared task, including legacy cache headers.

    Older materializers only wrote entity_table when label availability was
    explicit. The root role exists independently of that optional clock.
    Resolve it from the caller's task context, never a registry or name guess.
    The supervising portable entry point verifies the full context payload.
    """
    path = os.environ.get(CONTEXT_ENV)
    if not path:
        raise ValueError("Generic staged readouts require an explicit task context")
    source = ContextTaskSource(path, verify=False)
    if (metadata["dataset"],metadata["task"]) != (source.manifest["dataset"],source.manifest["task"]):
        raise ValueError("Readout metadata belongs to a different task context")
    root = source.task.entity_table
    if metadata.get("entity_table") not in {None,root}:
        raise ValueError("Readout root differs from the declared task root")
    return {**metadata,"entity_table":root}


def fit_signal(spec, base, x, y, times, vx, *, origin, meta, seed, threads, root, raw):
    kind = "classification" if meta["metric"] == "roc_auc" else "regression"
    categorical = [i for i, c in enumerate(x) if c in meta["feature_schema"]["categorical_columns"]]
    if spec.kind == "text":
        mask, support = mature_support_mask(times, origin=origin,
            horizon=meta["label_horizon"], refit_frames=meta["validation_cutoffs"], policy=base,
            available_at=x.attrs.get("kurversc_label_available_at"))
        text = [x.columns[i] for i in categorical]
        head = BoundedTextReadout(task=kind, random_state=seed).fit(
            x.loc[mask, text], y.loc[mask], categorical=text, root_prefix=root)
        return head.predict_values(vx), {**head.audit, "support":support}
    if spec.kind != "support":
        raise ValueError("Only text and frame-ensemble candidates are supported")
    model, audit = fit_temporal_policy(x, y, times, origin=origin,
        label_horizon=meta["label_horizon"], refit_frames=meta["validation_cutoffs"],
        policy=challenger_policy(base, spec), task=kind, categorical=categorical,
        random_state=seed, row_limit=raw["row_limit"],
        model_params={**raw["model_params"], "thread_count":threads})
    prediction = values(model, vx, kind)
    del model
    return prediction, audit


def _case(args, raw, meta, policy, fold, seed):
    train, held, phase = phase_frames(args.task_root, raw, policy, fold)
    control = incumbent_evidence(args.task_root/"raw-audit", fold, seed, policy)
    np.testing.assert_array_equal(control["truth"], held[meta["target"]])
    np.testing.assert_array_equal(control["groups"], held.__event_entity)
    y, times = train[meta["target"]], pd.to_datetime(train.__event_cutoff)
    task = "classification" if meta["metric"] == "roc_auc" else "regression"
    vectors, audit, unavailable = {}, {}, {}
    projector = TargetDomainProjection().fit(y, times,
        origin=phase["fit_origin"], horizon=meta["label_horizon"])
    if task == "regression":
        for name, applicable in (("nonnegative",projector.nonnegative), ("integer_counts",projector.integer_counts)):
            if applicable:
                vectors[name] = projector.predict_values(control["prediction"], recipe=name)
    if phase["optional_fit_available"]:
        x, vx = prepared(train,meta["feature_schema"]), prepared(held,meta["feature_schema"])
        source_root = phase_source_root(meta["dataset"],phase="inner")
        root = meta["entity_table"]
        raw_heads = {}
        for spec in SIGNALS:
            try:
                cache_key = (spec.kind, spec.training_frames)
                if cache_key not in raw_heads:
                    sx, sy, st = x, y, times
                    if spec.training_frames:
                        support, _, _ = phase_frames(args.task_root,raw,policy,fold,
                            frames=spec.training_frames)
                        sx = prepared(support,meta["feature_schema"])
                        sy, st = support[meta["target"]],pd.to_datetime(support.__event_cutoff)
                    raw_heads[cache_key] = fit_signal(spec,policy,sx,sy,st,vx,
                        origin=phase["fit_origin"],meta=meta,seed=seed,
                        threads=args.threads,root=root,raw=raw)
                pred, report = raw_heads[cache_key]
                vectors[spec.name] = apply_challenger_weight(spec,control["prediction"],pred)
                audit[spec.name] = report
            except (ChallengerUnavailable,InsufficientPolicySupport) as error:
                unavailable[spec.name] = str(error)
        if task == "regression" and (np.asarray(y)>=0).all() and (np.asarray(y)==0).sum()>=20 and (np.asarray(y)>0).sum()>=20:
            try:
                pred, audit["zero_aware"] = fit_prediction(x,y,times,vx,
                    raw=raw,meta=meta,policy=policy,phase=phase,seed=seed,zero=True)
                for weight in WEIGHTS:
                    vectors[f"zero_aware_{weight:g}"] = (1-weight)*control["prediction"]+weight*pred
            except ValueError as error:
                if str(error)!="Nonnegative numeric targets with >=20 positive and zero observations required":
                    raise
                unavailable["zero_aware"] = str(error)
        try:
            blocks = object_payload(train,held,meta=meta,source_root=source_root,
                origin=phase["fit_origin"],output=args.output/f"fold{fold}")
            xx, vxx = [pd.concat([a,b],axis=1) for a,b in ((x,blocks[0]),(vx,blocks[1]))]
            pred, audit["objects"] = fit_prediction(xx,y,times,vxx,
                raw=raw,meta=meta,policy=policy,phase=phase,seed=seed)
            vectors["object_full"], vectors["object_half"] = pred,.5*(control["prediction"]+pred)
        except ChallengerUnavailable as error:
            unavailable["objects"] = str(error)
    else:
        unavailable["payload_learners"] = "global_optional_feature_cell_capacity"
    return control,vectors,dict(fold=fold,seed=seed,origin=phase["origin"],
        scores={n:_score(control["truth"],p,task) for n,p in {"incumbent":control["prediction"],**vectors}.items()},
        audit=audit,unavailable=unavailable)


def validate(args):
    temporal = json.loads((args.task_root/"selection/results.json").read_text())
    policy = TemporalFitPolicy(**temporal["selected_policy"])
    meta = resolve_task_metadata(json.loads((args.task_root/"selection/manifest.json").read_text())["metadata"])
    identity = dict(contract=CONTRACT,metadata=meta,policy=asdict(policy),
        temporal_selection_sha256=digest(args.task_root/"selection/results.json"),
        threads=args.threads,heldout_targets_used=False)
    freeze(args.output/"manifest.json",identity)
    saved = args.output/"selection.json"
    if saved.exists():
        return json.loads(saved.read_text()),meta
    raw_path = args.task_root/"raw-audit/manifest.json"
    if not raw_path.exists():
        selected = dict(selected="incumbent",decision=dict(selected="incumbent",
            reason="insufficient_temporal_evidence",candidates={}),policy=asdict(policy),records=[])
        dump(saved,selected)
        return selected,meta
    raw = json.loads(raw_path.read_text())
    evidence, records = [],[]
    for fold in range(len(raw["folds"])):
        for seed in CONTRACT["seeds"]:
            stem = args.output/f"fold{fold}/seed{seed}"
            arrays, report = stem.with_suffix(".npz"),stem.with_suffix(".json")
            if arrays.exists() and report.exists():
                record = json.loads(report.read_text())
                assert record["manifest_sha256"] == digest(args.output/"manifest.json")
                with np.load(arrays) as a:
                    control = {k:a[k].copy() for k in ("truth","groups","prediction")}
                    vectors = {k:a[k].copy() for k in a.files if k not in control}
            else:
                control,vectors,record = _case(args,raw,meta,policy,fold,seed)
                arrays.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(arrays,**control,**vectors)
                record["manifest_sha256"] = digest(args.output/"manifest.json")
                dump(report,record)
            for name,pred in {"incumbent":control["prediction"],**vectors}.items():
                evidence.append(dict(origin=record["origin"],seed=seed,policy=name,
                    truth=control["truth"],groups=control["groups"],prediction=pred))
            records.append(record)
            dump(args.output/"progress.json",dict(completed_cases=len(records),required_cases=3*len(raw["folds"]),latest=record))
            print(json.dumps(dict(phase="generic_candidate_case",fold=fold,seed=seed,scores=record["scores"])),flush=True)
            gc.collect()
    kind = "classification" if meta["metric"]=="roc_auc" else "regression"
    decision = select_temporal_policy(evidence,task=kind,minimum_origins=2,
        minimum_seeds=3,deployment_seed=0,alpha=.05,bootstrap_repeats=600)
    selected = dict(selected=decision["selected"],decision=decision,policy=asdict(policy),
        records=records,contract=CONTRACT,heldout_targets_used=False)
    dump(saved,selected)
    return selected,meta


def official(args, selected, meta):
    choice = selected["selected"]
    locked = digest(args.output/"selection.json")
    baseline_root = args.task_root/"official"
    baseline = json.loads((baseline_root/"results.json").read_text())
    with np.load(baseline_root/"predictions.npz") as a:
        keys,control = artifact_keys(a),a["candidate"].copy()
    if baseline.get("query_keys_sha256") != key_fingerprint(keys):
        raise ValueError("Native prediction keys changed")
    candidate = control.copy()
    if choice != "incumbent":
        policy = TemporalFitPolicy(**selected["policy"])
        spec = next((s for s in SIGNALS if s.name==choice),SignalChallenger("generic_readout","support"))
        source,outer,train,forecast = materialize_outer(args,meta,spec,policy)
        new_keys = masked_query_keys(forecast,entity="__event_entity",timestamp="__event_cutoff",target=meta["target"])
        control = align_prediction(control,source_keys=keys,query_keys=new_keys)
        keys = new_keys
        y,times = train[meta["target"]],pd.to_datetime(train.__event_cutoff)
        origin = pd.to_datetime(outer.eval_table.df[source.task.time_col]).min()
        if choice in {"nonnegative","integer_counts"}:
            projector = TargetDomainProjection().fit(y,times,origin=outer.cutoff,horizon=meta["label_horizon"])
            candidate = projector.predict_values(control,recipe=choice)
        else:
            x,vx = prepared(train,meta["feature_schema"]),prepared(forecast,meta["feature_schema"])
            raw = json.loads((args.task_root/"raw-audit/manifest.json").read_text())
            if choice in {s.name for s in SIGNALS}:
                pred,_ = fit_signal(spec,policy,x,y,times,vx,origin=origin,meta=meta,
                    seed=0,threads=args.threads,root=source.task.entity_table,raw=raw)
                candidate = apply_challenger_weight(spec,control,pred)
            elif choice.startswith("zero_aware_"):
                pred,_ = fit_prediction(x,y,times,vx,raw=raw,meta=meta,policy=policy,
                    phase=dict(fit_origin=str(outer.cutoff)),seed=0,zero=True)
                weight = float(choice.removeprefix("zero_aware_"))
                candidate = (1-weight)*control+weight*pred
            elif choice in {"object_full","object_half"}:
                blocks = object_payload(train,forecast,meta=meta,
                    source_root=phase_source_root(meta["dataset"],phase="outer"),origin=outer.cutoff,output=args.output/"official")
                xx,vxx = [pd.concat([a,b],axis=1) for a,b in ((x,blocks[0]),(vx,blocks[1]))]
                pred,_ = fit_prediction(xx,y,times,vxx,raw=raw,meta=meta,policy=policy,
                    phase=dict(fit_origin=str(outer.cutoff)),seed=0)
                candidate = pred if choice=="object_full" else .5*(control+pred)
            else:
                raise ValueError("Candidate is outside the frozen generic menu")
    if locked != digest(args.output/"selection.json"):
        raise ValueError("Candidate choice changed during final fitting")
    if candidate.shape != control.shape or not np.isfinite(candidate).all():
        raise ValueError("Incomplete or nonfinite predictions")
    np.savez_compressed(args.output/"predictions.npz",control=control,candidate=candidate,**key_arrays(keys))
    score = (baseline["official_test_scores"]["candidate"] if choice=="incumbent"
        else float(source.task.evaluate(candidate,target_table=None)[meta["metric"]]))
    result = dict(dataset=meta["dataset"],task=meta["task"],metric=meta["metric"],
        selected=choice,arm=choice,official_test_scores=dict(control=baseline["official_test_scores"]["candidate"],candidate=score),
        query_keys_sha256=key_fingerprint(keys),selection_sha256=locked,
        heldout_targets_used_for_selection=False,deployment_seed=0)
    dump(args.output/"results.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread budget required")
    args.output.mkdir(parents=True,exist_ok=True)
    selected,meta = validate(args)
    print(json.dumps(official(args,selected,meta)),flush=True)


if __name__=="__main__":
    main()
