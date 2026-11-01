"""Production-faithful finalist check across arbitrary temporal block sizes.

Keep the post-votes experiment's bounded two-graph/native-readout comparison.
Generalize to two or three available origins and multi-cutoff validation/refit
blocks. Capacity fallbacks are target-blind and deterministic, not score based.
No task names, historical winners or official scores enter the selector.
"""

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts import benchmark_production_finalists as previous
from scripts.generic_staged_helpers import freeze
from scripts.run_gap_followup_queue import digest, dump


MAXIMUM_FEATURE_CELLS = 1_000_000_000
MAXIMUM_AUDIT_MATRIX_CELLS = 3_000_000_000
CONTRACT = {**previous.CONTRACT, "origins": "up to three available origins",
    "minimum_origins":2, "multi_cutoff_observed_refit":True,
    "forecast_entity_time_identity":True,
    "maximum_optional_feature_cells":MAXIMUM_FEATURE_CELLS,
    "maximum_optional_audit_matrix_cells":MAXIMUM_AUDIT_MATRIX_CELLS,
    "capacity_fallback":"unchanged incumbent; no partial comparison"}


def compare_cases(incumbent, challenger, control, candidate):
    from kurversc.core import _rerank_objective, _select_temporal_rerank_trials
    folds = sorted({f for f, _ in control})
    expected = {(f, s) for f in folds for s in range(3)}
    if not 2 <= len(folds) <= 3 or set(control) != expected or set(candidate) != expected:
        raise ValueError("Complete two/three-origin, three-seed evidence required")
    if incumbent.metric != "mae" or challenger.metric != "mae":
        raise ValueError("The optional bounded finalist comparison is MAE-only")
    if challenger.feature_count * 2 > incumbent.feature_count:
        return dict(selected="incumbent", reason="no_drastic_complexity_reduction")
    decisions, safety, summaries = {}, True, {}
    for seed in range(3):
        records = []
        for name, trial, cases in (("incumbent", incumbent, control), ("challenger", challenger, candidate)):
            scores, errors, labels = [], [], []
            for fold in folds:
                key = fold, seed
                for field in ("truth", "groups", "forecast_cutoffs"):
                    np.testing.assert_array_equal(control[key][field], candidate[key][field])
                if control[key]["origin"] != candidate[key]["origin"]:
                    raise ValueError("Unmatched forecast origins")
                y, prediction = np.asarray(cases[key]["truth"]), np.asarray(cases[key]["prediction"])
                if not y.size or y.shape != prediction.shape or not np.isfinite(prediction).all():
                    raise ValueError("Finite complete aligned validation vectors required")
                loss = np.abs(y-prediction)
                if not np.isfinite(loss).all():
                    raise ValueError("Finite mature validation targets required")
                scores.append(float(loss.mean()))
                errors.append(float(loss.std(ddof=1)/np.sqrt(len(loss))) if len(loss)>1 else 0.)
                labels.append(cases[key]["origin"])
            obj, se = _rerank_objective("mae", float(np.mean(scores)), float(np.std(scores)), len(folds), .25)
            records.append(replace(trial, model=None, stage="rerank", validation_score=float(np.mean(scores)),
                objective_score=obj, validation_standard_error=se, cutoff_labels=tuple(labels),
                cutoff_scores=tuple(scores), cutoff_standard_errors=tuple(errors)))
            summaries[f"{name}_seed{seed}"] = dict(scores=scores, objective=obj, features=trial.feature_count)
        chosen = _select_temporal_rerank_trials(records, classification_negligible_gain=.002,
            regression_negligible_relative_gain=.005, complexity_uncertainty_multiplier=1.,
            complexity_selector=True, practical_recent_ties=True)[-1]
        decisions[str(seed)] = "challenger" if chosen.config==challenger.config else "incumbent"
        safety = safety and all(b<=a*1.01+1e-12 for a,b in zip(records[0].cutoff_scores, records[1].cutoff_scores))
    admitted = safety and all(v=="challenger" for v in decisions.values())
    return dict(selected="challenger" if admitted else "incumbent",
        reason="unanimous_production_fidelity_complexity_tie" if admitted else "incumbent_retained",
        seed_decisions=decisions, all_seed_fold_practical_safety=safety, evidence=summaries,
        origins=len(folds), official_targets_used_for_selection=False)


def audit_manifest(root):
    raw = json.loads((Path(root)/"raw-audit/manifest.json").read_text())
    if (not 2<=len(raw["folds"])<=3 or raw["seeds"]!=[0,1,2] or raw["observed_refit_frames"]<1
            or not raw["metadata"]["atomic_date_replay"] or raw["metadata"]["fold"]!="current"):
        raise ValueError("Matched atomic two/three-origin, three-seed production lifecycle required")
    return raw


def read_cases(root):
    root=Path(root)
    raw=audit_manifest(root)
    result={}
    for fold, definition in enumerate(raw["folds"]):
        snapshot=pd.read_parquet(root/f"snapshots/fold{fold}/validation.parquet",
            columns=[raw["metadata"]["target"], "__event_entity", "__event_cutoff"])
        for seed in range(3):
            with np.load(root/f"raw-audit/fold{fold}-incumbent-seed{seed}.npz") as a:
                case={k:a[k].copy() for k in ("truth","groups","prediction")}
            np.testing.assert_array_equal(case["truth"],snapshot[raw["metadata"]["target"]])
            np.testing.assert_array_equal(case["groups"],snapshot.__event_entity)
            case.update(origin=definition["origin"],forecast_cutoffs=pd.to_datetime(snapshot.__event_cutoff).to_numpy())
            result[fold,seed]=case
    return result


def compatible_audits(left,right):
    a,b=audit_manifest(left),audit_manifest(right)
    for field in ("folds","seeds","row_limit","model_params","observed_refit_frames"):
        if a[field]!=b[field]:
            raise ValueError(f"Finalist native procedures differ: {field}")
    for field in ("dataset","task","target","metric","label_horizon","validation_cutoffs"):
        if a["metadata"][field]!=b["metadata"][field]:
            raise ValueError(f"Finalist forecast definitions differ: {field}")


def independent_repeat(root):
    from kurversc.temporal_policy import TemporalFitPolicy
    from scripts.generic_staged_helpers import phase_frames, prepared, fit_prediction
    root=Path(root)
    raw=audit_manifest(root)
    last=len(raw["folds"])-1
    train,held,phase=phase_frames(root,raw,TemporalFitPolicy("incumbent"),last,
        maximum_feature_cells=MAXIMUM_FEATURE_CELLS)
    if not phase["optional_fit_available"]:
        raise ValueError("Native replay exceeds the declared optional capacity")
    meta,schema=raw["metadata"],raw["metadata"]["feature_schema"]
    prediction,audit=fit_prediction(prepared(train,schema),train[meta["target"]],
        pd.to_datetime(train.__event_cutoff),prepared(held,schema),raw=raw,meta=meta,
        policy=TemporalFitPolicy("incumbent"),phase=phase,seed=0)
    with np.load(root/f"raw-audit/fold{last}-incumbent-seed0.npz") as a:
        np.testing.assert_array_equal(a["truth"],held[meta["target"]])
        np.testing.assert_array_equal(a["groups"],held.__event_entity)
        np.testing.assert_allclose(prediction,a["prediction"],atol=1e-9,rtol=0)
        error=float(np.max(np.abs(prediction-a["prediction"])))
    dump(root/"independent-repeat.json",dict(maximum_error=error,serialized_predictions_exact=True,audit=audit))
    del train,held,prediction
    gc.collect()


def planned_capacity(pair):
    # Use only the current complete full-data rerank's row counts/schema sizes.
    # This deliberately conservative, deterministic screen precedes extra SQL.
    estimates={name:int(t.train_rows+t.validation_rows)*int(t.feature_count)
        for name,(_,t) in zip(("incumbent","challenger"),pair)}
    return dict(estimated_cells=estimates,maximum_cells=MAXIMUM_FEATURE_CELLS,
        eligible=all(v<=MAXIMUM_FEATURE_CELLS for v in estimates.values()))


def materialized_capacity(root):
    """Check actual native support using only clocks, row counts and schema."""
    from kurversc.temporal_policy import TemporalFitPolicy, policy_support
    root=Path(root)
    meta=json.loads((root/"cache/materialization.json").read_text())
    clocks=pd.read_parquet(root/"cache/history.parquet",columns=["__event_cutoff"])
    width=len(meta["feature_schema"]["columns"])
    validation_rows=pq.ParquetFile(root/"cache/validation.parquet").metadata.num_rows
    matrix_cells=(len(clocks)+validation_rows)*width
    phases=[]
    for path in sorted((root/"snapshots").glob("fold*/manifest.json")):
        snapshot=json.loads(path.read_text())
        support=policy_support(clocks,timestamp="__event_cutoff",origin=snapshot["snapshot_cutoff"],
            label_horizon=meta["label_horizon"],refit_frames=meta["validation_cutoffs"],
            policy=TemporalFitPolicy("incumbent"))
        selected=pd.to_datetime(clocks.__event_cutoff).isin(pd.to_datetime(support["cutoffs"]))
        held=pq.ParquetFile(path.parent/"validation.parquet").metadata.num_rows
        phases.append((int(selected.sum())+held)*width)
    return dict(eligible=2<=len(phases)<=3 and matrix_cells<=MAXIMUM_AUDIT_MATRIX_CELLS
        and all(v<=MAXIMUM_FEATURE_CELLS for v in phases),audit_matrix_cells=matrix_cells,
        phase_cells=phases,observed_refit_frames=meta["validation_cutoffs"],target_values_accessed=False)


def run(args):
    from dataclasses import asdict
    from scripts.run_fresh_cloud_campaign import ROOT
    from scripts.run_kurversc_local_campaign import _run_guarded
    started=time.monotonic()
    pair=previous.choose_pair(args.graph_root,args.search_result,args.dataset,args.task)
    capacity=planned_capacity(pair) if pair else None
    manifest=dict(contract=CONTRACT,dataset=args.dataset,task=args.task,search_result=str(args.search_result),
        search_result_sha256=digest(args.search_result),capacity=capacity,source_sha256=digest(__file__),
        candidates=[] if pair is None else [dict(arm=name,checkpoint=str(p),checkpoint_sha256=digest(p),
            config=asdict(t.config),feature_count=t.feature_count) for name,(p,t) in zip(("incumbent","challenger"),pair)])
    freeze(args.output/"manifest.json",manifest)
    if pair is None or not capacity["eligible"]:
        dump(args.output/"results.json",dict(selected="incumbent",reason="no_bounded_alternative" if pair is None
            else "optional_feature_cell_capacity",official_targets_used_for_selection=False))
        return
    for candidate in manifest["candidates"]:
        arm=args.output/"arms"/candidate["arm"]
        freeze(arm/"plan.json",candidate)
        stages=(("materialize",arm/"task/cache/materialization.json"),
            ("snapshots",arm/"task/snapshots/manifest.json"),
            ("native",arm/"task/raw-audit/results.json"),("repeat",arm/"task/independent-repeat.json"))
        for name,completion in stages:
            if completion.exists():continue
            if digest(candidate["checkpoint"])!=candidate["checkpoint_sha256"]:
                raise ValueError("Fresh finalist checkpoint changed")
            dump(args.output/"progress.json",dict(active=candidate["arm"],stage=name))
            command=[sys.executable,"-u","-m","scripts.benchmark_production_finalists_v2","--mode","arm",
                "--stage",name,"--arm",candidate["arm"],"--graph-root",str(args.graph_root),
                "--search-result",str(args.search_result),"--dataset",args.dataset,"--task",args.task,"--output",str(args.output)]
            with (arm/f"{name}.log").open("a",buffering=1) as log:
                code,usage=_run_guarded(command,cwd=ROOT,environment=__import__("os").environ.copy(),log=log,
                    minimum_available_memory_gb=0,poll_seconds=5,
                    task_time_limit_seconds=max(1,args.hours*3600-(time.monotonic()-started)))
            dump(arm/f"{name}-resources.json",usage)
            if code or not completion.exists():
                raise RuntimeError(f"Production finalist {candidate['arm']}/{name} incomplete: {code}")
            if name=="snapshots":
                capacity=materialized_capacity(arm/"task")
                dump(arm/"actual-capacity.json",capacity)
                if not capacity["eligible"]:
                    dump(args.output/"results.json",dict(selected="incumbent",
                        reason="optional_actual_capacity_or_temporal_support",capacity=capacity,
                        official_targets_used_for_selection=False))
                    return
    roots=[args.output/f"arms/{name}/task" for name in ("incumbent","challenger")]
    compatible_audits(*roots)
    result=compare_cases(pair[0][1],pair[1][1],*(read_cases(r) for r in roots))
    chosen=manifest["candidates"][result["selected"]=="challenger"]
    dump(args.output/"results.json",dict(**result,checkpoint=chosen["checkpoint"],
        checkpoint_sha256=chosen["checkpoint_sha256"],config=chosen["config"],
        elapsed_seconds=time.monotonic()-started,manifest_sha256=digest(args.output/"manifest.json")))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("run","arm"),default="run")
    parser.add_argument("--stage",choices=("materialize","snapshots","native","repeat"))
    parser.add_argument("--arm",choices=("incumbent","challenger"))
    parser.add_argument("--graph-root",type=Path,required=True)
    parser.add_argument("--search-result",type=Path,required=True)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--task",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--hours",type=float,default=4.)
    args=parser.parse_args()
    if not 0<args.hours<24:parser.error("Positive <24h audit required")
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=="arm":
        if args.arm is None or args.stage is None:parser.error("Explicit arm/stage required")
        if args.stage=="repeat":independent_repeat(args.output/f"arms/{args.arm}/task")
        else:previous.arm_stage(args)
    else:run(args)


if __name__=="__main__":main()
