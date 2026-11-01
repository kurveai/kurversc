"""Shared mechanics for the generic staged candidate menu.

No exposure-role inference, semantic correction, recurrence recovery, sequence
recovery or bounded-view procedures are imported by this module.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import kurversc.core as core
from kurversc import temporal_policy
from kurversc.modeling import prepare_prediction_features
from kurversc.signal_challengers import (BoundedTextReadout, ChallengerUnavailable,
    challenger_policy, combine_predictions, mature_support_mask)
from kurversc.temporal_policy import (TemporalFitPolicy, InsufficientPolicySupport,
    fit_temporal_policy, policy_support)
from scripts.activity_timeline_challenger import table_metadata
from scripts.phase_frozen_object_breadth import PhaseFrozenObjectBreadth as ObjectBreadth
from scripts.zero_aware_quantile_readout import ZeroAwareQuantileReadout
MAX_OPTIONAL_FEATURE_CELLS = 500_000_000

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(path, value):
    path=Path(path)
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(value,default=str)):
        raise ValueError('Immutable generic-stage inputs changed')
    dump(path,value)

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def prepared(frame, schema):
    return prepare_prediction_features(frame, columns=list(schema["columns"]),
        categorical_columns=set(schema["categorical_columns"]),
        datetime_columns=set(schema["datetime_columns"]))


def routing_headers(frame, schema):
    """Own the nonpayload columns without retaining a wide numeric buffer."""
    payload = set(schema["columns"])
    return frame.loc[:, [name for name in frame if name not in payload]].copy()


def values(model, features, task):
    return np.asarray(model.predict_proba(features)[:, 1] if task == "classification"
                      else model.predict(features)).reshape(-1)


def incumbent_evidence(raw, fold_number, seed, policy):
    name = policy.name.split("__blend_")[0]
    path = raw / f"fold{fold_number}-{name}-seed{seed}.npz"
    with np.load(path) as arrays:
        row = {key: arrays[key].copy() for key in ("truth", "prediction", "groups")}
    if policy.blend_weight < 1:
        with np.load(raw / f"fold{fold_number}-incumbent-seed{seed}.npz") as base:
            np.testing.assert_array_equal(base["truth"], row["truth"])
            np.testing.assert_array_equal(base["groups"], row["groups"])
            row["prediction"] = combine_predictions(base["prediction"], row["prediction"], policy.blend_weight)
    return row


def cache_locations(task_root):
    """Prefer colocated artifacts; legacy manifests may reference shared caches."""
    manifest = json.loads((task_root / "selection/manifest.json").read_text())
    cache = task_root / "cache"
    snapshots = task_root / "snapshots"
    if not (cache / "history.parquet").exists():
        cache = Path(manifest["cache"])
    if not snapshots.exists():
        snapshots = Path(manifest["phase_frozen_cache"])
    return cache, snapshots


@contextmanager
def _capture(columns, target, entity, root):
    original = core._materialize_at_cutoff
    frames, cutoffs = [], []
    active = 0

    def materialize(*args, **kwargs):
        nonlocal active
        active += 1
        try:
            frame = original(*args, **kwargs)
            outermost = active == 1
        finally:
            active -= 1
        if not outermost:
            return frame
        key = entity if entity in frame else f"{root}_{entity}"
        if key not in frame:
            raise ValueError("Frozen frame lacks its entity key")
        missing = set([*columns, target]) - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"Frozen feature schema missing {len(missing)} columns at "
                f"{kwargs['cut_date']}; materialized rows={len(frame)}; "
                f"examples={sorted(missing)[:5]}"
            )
        frozen = frame.loc[:, [*columns, target]].copy()
        spec = args[6] if len(args)>6 else kwargs.get("label_spec")
        if spec is not None and getattr(spec,"available_at",None):
            frozen["__kurversc_label_available_at"] = frame[spec.available_at].to_numpy()
        frozen["__event_entity"] = frame[key].to_numpy()
        frozen["__event_cutoff"] = str(pd.Timestamp(kwargs["cut_date"]))
        frames.append(frozen)
        cutoffs.append(str(pd.Timestamp(kwargs["cut_date"])))
        return frame

    core._materialize_at_cutoff = materialize
    try:
        yield frames, cutoffs
    finally:
        core._materialize_at_cutoff = original


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


def task_root_table(metadata, source_meta):
    """Resolve RelBench's task-parquet placeholder using the declared task.

    Some task parquet metadata points to 'entity_table', not the physical DB
    table. Never guess a relation from task names or key values.
    """
    entity, root = next(iter(source_meta["fkey_col_to_pkey_table"].items()))
    declared = metadata.get("entity_table")
    if declared:
        if root not in {"entity_table",declared}:
            raise ValueError("Task context and source relation disagree about the predictive root")
        return declared
    if root == "entity_table":
        from relbench.tasks import get_task
        task = get_task(metadata["dataset"], metadata["task"], download=False)
        if task.entity_col != entity:
            raise ValueError("Task declaration and parquet root key differ")
        root = task.entity_table
    return root


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


def phase_frames(root, raw, policy, fold, *, frames=None,
                 maximum_feature_cells=MAX_OPTIONAL_FEATURE_CELLS, maximum_rows=None):
    cache, snapshots = cache_locations(root)
    meta = raw["metadata"]
    snapshot = json.loads((snapshots / f"fold{fold}/manifest.json").read_text())
    origin = snapshot["snapshot_cutoff"]
    if pd.Timestamp(snapshot["latest_observed_label_complete_at"]) > pd.Timestamp(origin):
        raise ValueError("Immature targets in a frozen source phase")
    availability = meta.get("label_available_at")
    headers = pd.read_parquet(cache / "history.parquet", columns=["__event_cutoff",*([availability] if availability else [])])
    support_policy = policy if frames is None else TemporalFitPolicy("three_initial", training_frames=frames)
    support = policy_support(headers, timestamp="__event_cutoff", origin=origin,
        label_horizon=meta["label_horizon"], refit_frames=meta["validation_cutoffs"], policy=support_policy)
    # Keep the on-disk timestamp representation (some old caches use strings).
    visible = pd.to_datetime(headers.__event_cutoff).isin(pd.to_datetime(support["cutoffs"]))
    count = int(visible.sum())
    if maximum_rows is not None and count > maximum_rows:
        raise ChallengerUnavailable(f"Optional mature support has {count} rows; capacity is {maximum_rows}")
    held_path = snapshots / f"fold{fold}/validation.parquet"
    cells = (count + pq.ParquetFile(held_path).metadata.num_rows) * len(meta["feature_schema"]["columns"])
    fit_available = cells <= maximum_feature_cells
    if frames is not None and not fit_available:
        raise ChallengerUnavailable(f"Optional learner payload has {cells} cells; capacity is {maximum_feature_cells}")
    columns = [*(meta["feature_schema"]["columns"] if fit_available else []),
               meta["target"], "__event_entity", "__event_cutoff",*([availability] if availability else [])]
    filter_cutoffs = headers.loc[visible, "__event_cutoff"].drop_duplicates().tolist()
    train = pd.read_parquet(cache / "history.parquet", columns=columns,
                           filters=[("__event_cutoff", "in", filter_cutoffs)])
    if availability:
        dates = pd.to_datetime(train.__event_cutoff)
        boundary = support["refit"][0] if support["refit"] else pd.Timestamp(origin)
        ready = pd.to_datetime(train[availability])
        train = train.loc[ready.le(pd.Timestamp(origin)) & (~dates.isin(support["initial"]) | ready.le(boundary))]
    held = pd.read_parquet(held_path, columns=columns)
    if (pd.to_datetime(train.__event_cutoff) + pd.Timedelta(meta["label_horizon"]) > pd.Timestamp(origin)).any():
        raise ValueError("Immature readout training observations")
    return train.reset_index(drop=True), held.reset_index(drop=True), dict(origin=snapshot["origin"], fit_origin=origin,
        optional_fit_available=fit_available, optional_feature_cells=cells)


def object_payload(train, held, *, meta, source_root, origin, output):
    source_meta = table_metadata(source_root / "tasks" / meta["task"] / "train.parquet")
    if len(source_meta["fkey_col_to_pkey_table"] or {}) != 1:
        raise ChallengerUnavailable("Single predictive root required for object history")
    root = task_root_table(meta, source_meta)
    try:
        provider = ObjectBreadth(source_root / "db", root, horizon=meta["label_horizon"],
            training_cutoff=pd.to_datetime(train.__event_cutoff).min(), temporary=output / "object-spill", threads=2)
    except ValueError as error:
        if "No training-visible high-cardinality object foreign key" not in str(error):
            raise
        raise ChallengerUnavailable(str(error)) from error
    try:
        return [provider.transform(f, phase_boundary=origin) for f in (train, held)]
    finally:
        provider.close()


def materialize_outer(args, metadata, spec, base_policy):
    """Frozen graph materialization, with one-tree drivers whose scores are discarded."""
    import kurversc
    from kurversc.plan_dates import atomic_execution_plan
    from kurversc.training_support import available_root_training, select_training_support
    from kurversc.staged_adapter import resolve_cache_config
    from kurversc.task_source import load_task_source as RelBenchDatasetTask
    from kurversc.staged_adapter import KURVERSC_DEFAULTS, _SelectedGraph, _runtime_config
    from scripts.run_kurversc_selected_relarena import AuditWinnerSystem

    checkpoint = Path(metadata["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
        raise ValueError("Graph checkpoint changed after validation")
    trial = pickle.loads(checkpoint.read_bytes())["trial"]
    trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    source = RelBenchDatasetTask(metadata["dataset"], metadata["task"], download=False)
    outer, task = source.outer_split(), source.task
    KURVERSC_DEFAULTS.update(full_training_frames=1, full_validation_frames=None,
        base_distribution_features=True, temporal_estimator_mode="frame_ensemble",
        duckdb_memory_limit=None, duckdb_max_temp_directory_size=None,
        model_params={"iterations":1, "thread_count":1})
    system = AuditWinnerSystem(cache=resolve_cache_config(None, on_miss="raise"), run_identity=source.run_identity())
    system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
    _, problem = system._fit_arm(task, outer.db_state, outer.train_table, outer.val_table,
        seed=0, phase="outer", selection=_SelectedGraph(config=trial.config,
            execution_plan=copy.deepcopy(trial.execution_plan)))
    root = outer.db_state.table_dict[task.entity_table]
    policy = challenger_policy(base_policy, spec)
    available = available_root_training(outer.train_table.df, root.df, label_key=task.entity_col,
        root_key=root.pkey_col, timestamp=task.time_col, root_timestamp=root.time_col)
    support = select_training_support(available, timestamp=task.time_col,
        evaluation_cutoff=pd.to_datetime(outer.val_table.df[task.time_col]).min(),
        label_horizon=task.timedelta, minimum_rows=policy.minimum_training_rows,
        minimum_frames=policy.training_frames, max_frames=policy.max_training_frames)
    if not support.complete or len(support.cutoffs) < policy.training_frames:
        raise InsufficientPolicySupport("Final training support is incomplete")
    labels = problem.label_node
    history = outer.train_table.df.loc[pd.to_datetime(outer.train_table.df[task.time_col]).isin(support.cutoffs)].copy()
    history[labels.split] = labels.train_value
    validation = labels.source.loc[labels.source[labels.split] != labels.train_value].copy()
    observed = pd.concat([history, validation], ignore_index=True)
    first_test = pd.to_datetime(outer.eval_table.df[task.time_col]).min()
    if (pd.to_datetime(observed[task.time_col]) + task.timedelta > first_test).any():
        raise ValueError("Immature final labels")
    problem = replace(problem, label_node=replace(labels, source=observed))
    options = _runtime_config()
    options.update(full_training_frames=len(support.cutoffs), temporal_estimator_mode="pooled",
        model_params={"iterations":1, "thread_count":1})
    schema = metadata["feature_schema"]
    with _capture(schema["columns"], task.target_col, task.entity_col, task.entity_table) as (frames, _):
        fitted = kurversc.fit(**problem.fit_kwargs(), **options, random_state=0,
            preselected_config=trial.config, preselected_execution_plan=copy.deepcopy(trial.execution_plan))
    train = pd.concat(frames, ignore_index=True)
    with capture_prediction(schema, task.entity_col, task.entity_table) as chunks:
        system._predict(task, outer.eval_table, result=fitted, problem=problem)
    forecast = pd.concat(chunks, ignore_index=True)
    if sorted(forecast["__order"].tolist()) != list(range(len(outer.eval_table.df))):
        raise ValueError("Masked test query order changed")
    forecast = forecast.sort_values("__order", kind="stable").reset_index(drop=True)
    del fitted
    gc.collect()
    return source, outer, train, forecast


