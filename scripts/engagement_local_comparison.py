"""Shared bounded, streamed inputs for two isolated local engagement tests."""

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from kurversc.modeling import fit_final_catboost
from kurversc.temporal_policy import TemporalFitPolicy, _score
from scripts.benchmark_signal_challengers import incumbent_evidence,prepared
from scripts.matched_learner_challengers import positioned_frame,sha256
from scripts.benchmark_support_transitions import checked_support_hash
from scripts.select_activity_readout import cache_locations
from scripts.run_gap_followup_queue import dump


TASK = Path("results/temporal-policy-all21-20261005/tasks/rel-stack_user-engagement")
SUPPORT = Path("results/user-engagement-recurrence-support3-v2-20261008")
MENU_ALLOWANCE = 104


def context(fold):
    manifest = json.loads((SUPPORT/"manifest.json").read_text())
    metadata = json.loads((TASK/"selection/manifest.json").read_text())["metadata"]
    if metadata!=manifest["metadata"] or json.loads((SUPPORT/"selection.json").read_text())["decision"]["selected"]!="three_initial":
        raise ValueError("Protected task/control identity changed")
    cache,snapshots = cache_locations(TASK)
    target,schema = metadata["target"],metadata["feature_schema"]
    routing = [target,"__event_entity","__event_cutoff"]
    headers = pd.read_parquet(cache/"history.parquet",columns=routing)
    held = pd.read_parquet(snapshots/f"fold{fold}/validation.parquet",columns=routing)
    phase = json.loads((snapshots/f"fold{fold}/manifest.json").read_text())
    if phase["snapshot_cutoff"]!=phase["origin"] or pd.Timestamp(phase["latest_observed_label_complete_at"])>pd.Timestamp(phase["origin"]):
        raise ValueError("Exact authorized validation phase required")
    audit = json.loads((SUPPORT/f"fold{fold}/seed0.json").read_text())["fit_audit"]
    cuts = audit["initial_cutoffs"]+audit["observed_refit_cutoffs"]
    positions = np.flatnonzero(pd.to_datetime(headers.__event_cutoff).isin(pd.to_datetime(cuts)).to_numpy())
    train = headers.iloc[positions].reset_index(drop=True)
    old = json.loads((SUPPORT/f"fold{fold}/support.json").read_text())
    checked_support_hash(train,target,old["ordered_train_hash"])
    if (pd.to_datetime(train.__event_cutoff)+pd.Timedelta(metadata["label_horizon"])>pd.Timestamp(phase["origin"])).any():
        raise ValueError("Immature training labels")
    legacy = TemporalFitPolicy(**json.loads((TASK/"selection/results.json").read_text())["selected_policy"])
    return dict(manifest=manifest,metadata=metadata,cache=cache,snapshots=snapshots,headers=headers,
        held=held,phase=phase,cuts=cuts,positions=positions,train=train,order=old,legacy=legacy,fold=fold)


def supplemental(ctx,split,positions=None):
    blocks = []
    for name in ("activity","label_state","recurrence"):
        path = SUPPORT/f"fold{ctx['fold']}"/f"{name}-{split}.parquet"
        columns = pq.ParquetFile(path).schema_arrow.names
        block = positioned_frame(path,columns,positions) if positions is not None else pd.read_parquet(path)
        blocks.append(block)
    return pd.concat(blocks,axis=1)


def protected(ctx,seed):
    legacy = incumbent_evidence(TASK/"raw-audit",ctx["fold"],seed,ctx["legacy"])
    with np.load(SUPPORT/f"fold{ctx['fold']}"/f"seed{seed}.npz") as arrays:
        control = arrays["candidate"].copy()
        for key,column in (("truth",ctx["metadata"]["target"]),("groups","__event_entity")):
            np.testing.assert_array_equal(arrays[key],ctx["held"][column])
            np.testing.assert_array_equal(legacy[key],arrays[key])
    return control,legacy["prediction"]


def original_frames(ctx):
    schema = ctx["metadata"]["feature_schema"]
    vx = prepared(positioned_frame(ctx["snapshots"]/f"fold{ctx['fold']}/validation.parquet",
        schema["columns"],np.arange(len(ctx["held"]))),schema)
    return pd.concat([vx,supplemental(ctx,"validation")],axis=1)


def streamed_head(ctx,seed,*,extra=None,vextra=None,graph_folder=None,progress=None):
    """Same four complete cohorts, 600-tree learner and exact column order."""
    schema = ctx["metadata"]["feature_schema"] if graph_folder is None else json.loads((graph_folder/"schema.json").read_text())
    if graph_folder is None:
        vx = original_frames(ctx)
    else:
        vx = prepared(pd.read_parquet(graph_folder/"validation.parquet",columns=schema["columns"]),schema)
        vx = pd.concat([vx,supplemental(ctx,"validation")],axis=1)
    if graph_folder is None and list(vx)!=ctx["order"]["columns"]:
        raise ValueError("Protected feature ordering changed")
    if vextra is not None:
        if len(vextra)!=len(vx):
            raise ValueError("Auxiliary held-out forecast rows changed")
        vx = pd.concat([vx,vextra],axis=1)
    categorical = [i for i,c in enumerate(vx) if c in schema["categorical_columns"]]
    vectors = []
    for number,cutoff in enumerate(ctx["cuts"]):
        selected = np.flatnonzero(pd.to_datetime(ctx["headers"].__event_cutoff).eq(pd.Timestamp(cutoff)).to_numpy())
        mapped = np.searchsorted(ctx["positions"],selected)
        np.testing.assert_array_equal(ctx["positions"][mapped],selected)
        if graph_folder is None:
            tx = prepared(positioned_frame(ctx["cache"]/"history.parquet",schema["columns"],selected),schema)
        else:
            table = pd.read_parquet(graph_folder/f"training-{number}.parquet")
            np.testing.assert_array_equal(table.__event_entity,ctx["headers"].iloc[selected].__event_entity)
            np.testing.assert_array_equal(table[ctx["metadata"]["target"]],ctx["headers"].iloc[selected][ctx["metadata"]["target"]])
            tx = prepared(table,schema)
            del table
        tx = pd.concat([tx,supplemental(ctx,"training",mapped)],axis=1)
        if extra is not None:
            if len(extra)!=len(ctx["positions"]):
                raise ValueError("Auxiliary training forecast rows changed")
            tx = pd.concat([tx,extra.iloc[mapped].reset_index(drop=True)],axis=1)
        if progress:
            progress(dict(phase="fit_strong_head",member=number,cutoff=cutoff,seed=seed,rows=len(tx),features=len(tx.columns)))
        model,_,_ = fit_final_catboost(tx,ctx["headers"].iloc[selected][ctx["metadata"]["target"]].reset_index(drop=True),
            task="classification",categorical=categorical,random_state=seed,model_params=ctx["manifest"]["model_params"])
        vectors.append(model.predict_proba(vx)[:,1])
        del tx,model
        gc.collect()
    return np.mean(vectors,axis=0)


def freeze(path,value):
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(value,default=str)):
        raise ValueError("Immutable local experiment changed")
    dump(path,value)


def source_identity(paths):
    return {str(p):sha256(p) for p in paths}


def report_case(path,ctx,seed,control,candidates,**details):
    truth,groups = ctx["held"][ctx["metadata"]["target"]].to_numpy(),ctx["held"].__event_entity.to_numpy()
    scores = {name:_score(truth,pred,"classification") for name,pred in {"incumbent":control,**candidates}.items()}
    np.savez_compressed(path.with_suffix(".npz"),truth=truth,groups=groups,incumbent=control,**candidates)
    report = dict(fold=ctx["fold"],seed=seed,origin=ctx["phase"]["origin"],scores=scores,
        gains={name:score-scores["incumbent"] for name,score in scores.items() if name!="incumbent"},**details)
    dump(path,report)
    print(json.dumps(report,default=str),flush=True)
    return report
