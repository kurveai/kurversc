"""Fixed official-protocol replay of a validation-supported one-off overlay.

Exploratory recovery: historical test results motivated this representation.
No core/default edits, no graph discovery, and no test-based recipe selection.
"""

import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

import kurversc.core as core
from kurversc.modeling import FrameEnsemble, prepare_prediction_features
from kurversc.staged_adapter import resolve_cache_config, KURVERSC_DEFAULTS, _SelectedGraph
from kurversc.task_source import load_task_source as RelBenchDatasetTask
from scripts.benchmark_avito_recovered_quality import capture_fit_inputs, dump, calibrated_features
from scripts.benchmark_frozen_event_summaries import _capture
from scripts.run_kurversc_selected_relarena import AuditWinnerSystem


@contextmanager
def capture_prediction(schema, entity, root):
    original=core._materialize_at_cutoff
    frames=[]
    def observed(*args,**kwargs):
        frame=original(*args,**kwargs)
        key=entity if entity in frame else f"{root}_{entity}"
        orders=[name for name in frame if name.startswith("__kurversc_prediction_order__")]
        if len(orders)!=1:
            raise ValueError("Prediction output has no unique row-order marker")
        saved=frame.loc[:,schema["columns"]].copy()
        saved["__event_entity"]=frame[key].to_numpy()
        saved["__event_cutoff"]=str(pd.Timestamp(kwargs["cut_date"]))
        saved["__order"]=frame[orders[0]].to_numpy()
        frames.append(saved)
        return frame
    core._materialize_at_cutoff=observed
    try:
        yield frames
    finally:
        core._materialize_at_cutoff=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--validation-panel",type=Path,required=True)
    parser.add_argument("--provider-script",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--quality-arm",choices=("quality_only","quality_no_activity","quality_calibrated","quality_regularized"),default="quality_only")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    # Freeze evidence/recipe before the official split or scorer is accessed.
    evidence={}
    for fold in ("prior","current"):
        path=args.validation_panel/f"historical-{fold}/results.json"
        payload=json.loads(path.read_text())
        expected=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if payload["manifest"]["checkpoint_sha256"]!=expected:
            raise ValueError("Validation used a different frozen historical recipe")
        rows={ (row["arm"],row["seed"]):row for row in payload["records"] }
        gains=[rows[(args.quality_arm,seed)]["score"]-rows[("control",seed)]["score"] for seed in (0,1,2)]
        if min(gains)<=0:
            raise ValueError("Quality overlay did not improve every validation seed")
        evidence[fold]=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gains=gains)
    trial=pickle.loads(args.checkpoint.read_bytes())["trial"]
    if trial.config.base_distribution_features or trial.config.include_target_history or trial.config.root_reachable_materialization:
        raise ValueError("Expected the exact frozen historical recovery control")
    schema=trial.execution_plan["kurversc_feature_schema"]
    added=("quality",) if args.quality_arm=="quality_no_activity" else ("activity","quality")
    if args.quality_arm=="quality_calibrated":
        added=(*added,"calibrated")
    manifest=dict(recipe="Frozen historical base+conditional, three initial frames plus observed validation refit",
        additional_blocks=added,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),validation=evidence,
        provider_sha256=hashlib.sha256(args.provider_script.read_bytes()).hexdigest(),seed=args.seed,
        scope="Exploratory official test diagnostic, not automatic benchmark/default promotion")
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/"manifest.json",manifest)
    source=RelBenchDatasetTask("rel-avito","user-clicks",download=False)
    task,outer=source.task,source.outer_split()
    KURVERSC_DEFAULTS.update(full_training_frames=3,search_training_frames=3,
        temporal_estimator_mode="frame_ensemble",base_distribution_features=None,
        duckdb_memory_limit=None,duckdb_max_temp_directory_size=None)
    AuditWinnerSystem.peer_context_mode=None
    AuditWinnerSystem.temporal_fit_policy="fixed"
    system=AuditWinnerSystem(cache=resolve_cache_config(None,on_miss="raise"),run_identity=source.run_identity())
    selection=_SelectedGraph(config=trial.config,execution_plan=copy.deepcopy(trial.execution_plan),training_frames=3)
    with capture_fit_inputs() as inputs, _capture(schema["columns"],task.target_col,task.entity_col,task.entity_table) as (frames,cuts):
        result,problem=system._fit_arm(task,outer.db_state,outer.train_table,outer.val_table,
            seed=args.seed,phase="outer",selection=selection)
    if len(inputs)!=4 or len(frames)!=4:
        raise ValueError("Official recovery must fit three initial plus one observed validation model")
    with capture_prediction(schema,task.entity_col,task.entity_table) as prediction_frames:
        control=system._predict(task,outer.eval_table,result=result,problem=problem)
    spec=importlib.util.spec_from_file_location("exposure_oneoff",args.provider_script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider=module.ExposureAdjustedSignals(outer.db_state.table_dict["SearchInfo"].df,
        outer.db_state.table_dict["SearchStream"].df,outer.db_state.table_dict["AdsInfo"].df,
        horizon=str(task.timedelta),threads=8,temp_directory=args.output/"spill")
    models,audits=[],[]
    try:
        for number,((x,y,kwargs,_),frame,cut) in enumerate(zip(inputs,frames,cuts,strict=True)):
            if len(x)!=len(frame):
                raise ValueError("Unexpected row sampling; overlay is not aligned")
            extras=provider.transform(frame["__event_entity"],min(pd.Timestamp(cut),pd.Timestamp(outer.cutoff)))
            for value in extras.values():
                value.index=x.index
            extras["calibrated"]=calibrated_features(extras)
            augmented=pd.concat([x,*[extras[name] for name in added]],axis=1)
            kwargs=copy.deepcopy(kwargs)
            if args.quality_arm=="quality_regularized":
                kwargs["model_params"]={**(kwargs["model_params"] or {}),"depth":5,"l2_leaf_reg":30}
            models.append(core._fit_temporal_frame_estimator(augmented,y,**kwargs)[0])
            audits.append(provider.last_audit)
            print(json.dumps(dict(event="quality_model_ready",frame=number,rows=len(x),features=len(augmented.columns))),flush=True)
        ensemble=FrameEnsemble(models,task="classification")
        chunks,orders=[],[]
        for frame in prediction_frames:
            extras=provider.transform(frame["__event_entity"],min(pd.Timestamp(frame["__event_cutoff"].iloc[0]),pd.Timestamp(outer.cutoff)))
            base=prepare_prediction_features(frame,columns=schema["columns"],categorical_columns=set(schema["categorical_columns"]),datetime_columns=set(schema["datetime_columns"]))
            for value in extras.values():
                value.index=base.index
            extras["calibrated"]=calibrated_features(extras)
            chunks.append(ensemble.predict_proba(pd.concat([base,*[extras[name] for name in added]],axis=1))[:,1])
            orders.append(frame["__order"].to_numpy())
            audits.append(provider.last_audit)
    finally:
        provider.close()
    order=np.concatenate(orders)
    if sorted(order.tolist())!=list(range(len(outer.eval_table.df))):
        raise ValueError("Masked test rows were not preserved exactly once")
    candidate=np.concatenate(chunks)[np.argsort(order,kind="stable")]
    np.savez_compressed(args.output/"masked-test-predictions.npz",control=control,candidate=candidate)
    # All feature/model decisions and predictions have now been fixed.
    control_score=float(task.evaluate(control,target_table=None)["roc_auc"])
    if args.seed==0 and abs(control_score-.6847007272635763)>1e-12:
        raise ValueError("Historical official control failed exact reproduction")
    candidate_score=float(task.evaluate(candidate,target_table=None)["roc_auc"])
    dump(args.output/"feature-audit.json",audits)
    dump(args.output/"results.json",dict(manifest=manifest,control_score=control_score,
        candidate_score=candidate_score,gain=candidate_score-control_score,
        training_models=len(models),feature_count=len(augmented.columns),
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event="complete",control_score=control_score,candidate_score=candidate_score,gain=candidate_score-control_score)),flush=True)


if __name__=="__main__":
    main()
