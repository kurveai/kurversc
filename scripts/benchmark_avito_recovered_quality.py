"""Validation-only ad-quality overlay on recovered frozen user-clicks graphs.

No test rows/scorer or production changes. Compare activity and quality against
the EXACT ordinary three-frame fit, capped by the history actually available at
each inner forecast origin. Features remain a one-off exposure prototype.
"""

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import resource
import sys
import time

import numpy as np
import pandas as pd

import kurversc.core as core
from kurversc.modeling import FrameEnsemble, fit_final_catboost, prepare_prediction_features, sample_training_rows
from kurversc.staged_adapter import resolve_cache_config, KURVERSC_DEFAULTS, _copy_timeless_tables, _drop_dangling_seeds
from kurversc.task_source import load_task_source as RelBenchDatasetTask
from scripts.benchmark_frozen_event_summaries import _capture, _paired_interval, _prediction_score
from scripts.run_kurversc_selected_relarena import AuditWinnerSystem


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def calibrated_features(extra):
    """Bounded opportunity-normalized response features; no target labels."""
    activity,quality=extra["activity"],extra["quality"]
    result={}
    for window in ("all","w0","w1","w2"):
        n=activity[f"exposure_activity_{window}_exposures"].clip(lower=0)
        clicks=activity[f"exposure_activity_{window}_clicks"].clip(lower=0)
        p=quality[f"exposure_quality_{window}_ad_expected_mean"].clip(0,1)
        slot=quality[f"exposure_quality_{window}_ad_position_expected_mean"].clip(0,1)
        expected=n*p
        result[f"calibrated_{window}_expected_ad_clicks"]=expected
        result[f"calibrated_{window}_expected_position_clicks"]=n*slot
        result[f"calibrated_{window}_clicks_per_expected_ad"]=(clicks+.5)/(expected+.5)
        result[f"calibrated_{window}_excess_click_z_ad"]=(clicks-expected)/np.sqrt(1+n*p*(1-p))
    return pd.DataFrame(result,index=activity.index).astype("float32")


@contextmanager
def capture_fit_inputs():
    """Observe exact prepared estimator inputs, without modifying core code."""
    original = core._fit_temporal_frame_estimator
    captured = []

    def observed(features, labels, **kwargs):
        fitted = original(features, labels, **kwargs)
        captured.append((features.copy(deep=True), labels.copy(deep=True),
                         copy.deepcopy(kwargs), fitted[0].get_params()))
        return fitted

    core._fit_temporal_frame_estimator = observed
    try:
        yield captured
    finally:
        core._fit_temporal_frame_estimator = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--peer-mode", choices=("off", "both"), required=True)
    parser.add_argument("--fold", choices=("current", "prior"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-script", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--arms", default="control,activity,quality_only")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    trial = pickle.loads(args.checkpoint.read_bytes())["trial"]
    if not trial.execution_plan or trial.status != "completed":
        raise ValueError("A completed trusted frozen trial is required")
    schema = trial.execution_plan["kurversc_feature_schema"]
    source = RelBenchDatasetTask("rel-avito", "user-clicks", download=False)
    task, inner = source.task, source.inner_split()
    if args.fold == "current":
        train, validation, db, boundary = inner.train_table, inner.eval_table, inner.db_state, inner.cutoff
    else:
        boundary = pd.Timestamp(inner.train_table.df[task.time_col].max())
        train, validation = copy.copy(inner.train_table), copy.copy(inner.train_table)
        timestamps = pd.to_datetime(inner.train_table.df[task.time_col])
        train.df = inner.train_table.df.loc[(timestamps < boundary) & (timestamps + task.timedelta <= boundary)].copy()
        validation.df = inner.train_table.df.loc[timestamps == boundary].copy()
        db = inner.db_state.upto(boundary)
        _copy_timeless_tables(db)
        source._dataset.validate_and_correct_db(db)
        count = len(db.table_dict[task.entity_table].df)
        train = _drop_dangling_seeds(train, task.entity_col, count)
        validation = _drop_dangling_seeds(validation, task.entity_col, count)
    KURVERSC_DEFAULTS.update(full_training_frames=3, full_validation_frames=None,
                            temporal_estimator_mode="frame_ensemble", base_distribution_features=None,
                            duckdb_memory_limit=None, duckdb_max_temp_directory_size=None)
    AuditWinnerSystem.profile = "fixed-config"
    AuditWinnerSystem.fixed_config = trial.config
    AuditWinnerSystem.fixed_execution_plan = copy.deepcopy(trial.execution_plan)
    AuditWinnerSystem.fixed_search_training_frames = 3
    AuditWinnerSystem.full_training_frames = 3
    AuditWinnerSystem.temporal_fit_policy = "fixed"
    AuditWinnerSystem.root_reachable_materialization = trial.config.root_reachable_materialization
    AuditWinnerSystem.peer_context_mode = None if args.peer_mode == "off" else args.peer_mode
    AuditWinnerSystem.peer_max_history_rows = 50_000
    system = AuditWinnerSystem(cache=resolve_cache_config(None, on_miss="raise"), run_identity=source.run_identity())
    with capture_fit_inputs() as inputs, _capture(schema["columns"], task.target_col, task.entity_col, task.entity_table) as (frames, cuts):
        fitted, problem = system._fit_arm(task, db, train, validation, seed=0, phase="inner")
    if len(frames) != fitted.fitted_model.training_frames + 1:
        raise RuntimeError("Expected exactly the materialized training frames plus one heldout frame")
    truth = pd.to_numeric(frames[-1][task.target_col]).to_numpy()
    vx = prepare_prediction_features(frames[-1], columns=list(schema["columns"]),
            categorical_columns=set(schema["categorical_columns"]), datetime_columns=set(schema["datetime_columns"]))
    reference = fitted.fitted_model.validation_estimator.predict_proba(vx)[:, 1]
    reference_score = _prediction_score(truth, reference, "classification")
    if abs(reference_score - fitted.full_validation_score) > 1e-12:
        raise RuntimeError("Captured validation rows do not reproduce the ordinary fit")
    if len(inputs) != fitted.fitted_model.training_frames + 1:
        raise RuntimeError("Unexpected estimator lifecycle; refusing a nonmatched control")
    prepared = []
    input_audit = []
    for number, ((features, labels, kwargs, params), frame) in enumerate(zip(inputs[:-1], frames[:-1], strict=True)):
        raw = prepare_prediction_features(frame, columns=list(schema["columns"]),
            categorical_columns=set(schema["categorical_columns"]), datetime_columns=set(schema["datetime_columns"]))
        pd.testing.assert_frame_equal(features, raw)
        pd.testing.assert_series_equal(labels, frame[task.target_col])
        prepared.append(features)
        input_audit.append(dict(frame=number, parameters=params, call=kwargs,
                               rows=len(features), dtypes={name:str(dtype) for name,dtype in features.dtypes.items()}))
    dump(args.output/"exact-input-audit.json",input_audit)
    manifest = dict(scope="Validation-only one-off; no test scoring and no production changes",
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), config=asdict(trial.config),
        peer_mode=args.peer_mode, fold=args.fold, boundary=str(boundary), cutoffs=cuts,
        requested_training_frames=3, actual_training_frames=fitted.fitted_model.training_frames,
        rows=[len(frame) for frame in frames], control_seed0_score=reference_score,
        seeds=[0,1,2], threads=args.threads, quality_provider_sha256=hashlib.sha256(args.provider_script.read_bytes()).hexdigest())
    dump(args.output/"manifest.json", manifest)
    del fitted, problem
    gc.collect()
    spec = importlib.util.spec_from_file_location("isolated_exposure_provider", args.provider_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = module.ExposureAdjustedSignals(db.table_dict["SearchInfo"].df,
        db.table_dict["SearchStream"].df, db.table_dict["AdsInfo"].df,
        horizon=str(task.timedelta), threads=args.threads, temp_directory=args.output/"spill")
    extras, feature_audit = [], []
    try:
        for number, (frame, cut) in enumerate(zip(frames, cuts, strict=True)):
            supplied = min(pd.Timestamp(cut), pd.Timestamp(boundary))
            extra = provider.transform(frame["__event_entity"], supplied)
            for value in extra.values():
                value.index = frame.index
            extra["calibrated"] = calibrated_features(extra)
            extras.append(extra)
            feature_audit.append(provider.last_audit)
            print(json.dumps(dict(event="quality_features_ready", frame=number, cutoff=cut, rows=len(frame))), flush=True)
    finally:
        provider.close()
    dump(args.output/"feature-audit.json", feature_audit)
    base_features = [*prepared, vx]
    categorical = [i for i, column in enumerate(schema["columns"]) if column in set(schema["categorical_columns"])]
    available_arms = {"control": (), "activity": ("activity",), "quality_only": ("activity", "quality"),
                      "quality_no_activity": ("quality",),
                      "quality_calibrated": ("activity","quality","calibrated"),
                      "quality_regularized": ("activity","quality")}
    requested = args.arms.split(",")
    if requested[0] != "control" or len(set(requested)) != len(requested) or set(requested)-set(available_arms):
        raise ValueError("A bounded unique arm list starting with the exact control is required")
    arms = {name: available_arms[name] for name in requested}
    manifest["arms"] = list(arms)
    dump(args.output/"manifest.json",manifest)
    predictions, records = {}, []
    for arm, added in arms.items():
        arm_x = [pd.concat([base, *[extra[name] for name in added]], axis=1) for base, extra in zip(base_features,extras, strict=True)]
        for seed in (0,1,2):
            models = []
            for number, (x, frame) in enumerate(zip(arm_x[:-1], frames[:-1], strict=True), start=1):
                tx, ty = sample_training_rows(x, frame[task.target_col], limit=100_000,
                    task="classification", random_state=seed+number)
                fit_kwargs = copy.deepcopy(inputs[number-1][2])
                fit_kwargs["random_state"] = seed
                fit_kwargs["model_params"] = {**(fit_kwargs["model_params"] or {}), "thread_count":args.threads}
                if arm == "quality_regularized":
                    fit_kwargs["model_params"].update(depth=5,l2_leaf_reg=30)
                models.append(core._fit_temporal_frame_estimator(tx, ty, **fit_kwargs)[0])
            prediction = FrameEnsemble(models, task="classification").predict_proba(arm_x[-1])[:,1]
            if arm=="control" and seed==0:
                np.testing.assert_allclose(prediction, reference, rtol=0, atol=1e-10)
            predictions[(arm,seed)] = prediction
            np.savez_compressed(args.output/f"{arm}-seed{seed}.npz", truth=truth, prediction=prediction,
                                groups=frames[-1]["__event_entity"].to_numpy())
            row = dict(arm=arm, seed=seed, score=_prediction_score(truth,prediction,"classification"),
                       feature_count=len(arm_x[-1].columns), actual_training_frames=len(models))
            records.append(row)
            dump(args.output/"progress.json", dict(records=records))
            print(json.dumps(row), flush=True)
            del models
            gc.collect()
    control = np.mean([predictions[("control",seed)] for seed in (0,1,2)], axis=0)
    comparisons = []
    for arm in arms:
        if arm == "control":
            continue
        candidate = np.mean([predictions[(arm,seed)] for seed in (0,1,2)], axis=0)
        comparisons.append(dict(candidate=arm, control_score=_prediction_score(truth,control,"classification"),
            candidate_score=_prediction_score(truth,candidate,"classification"),
            gain=_prediction_score(truth,candidate,"classification")-_prediction_score(truth,control,"classification"),
            paired_row_bootstrap_95pct=_paired_interval(truth,control,candidate,"classification")))
    payload = dict(manifest=manifest, records=records, comparisons=comparisons,
        elapsed_seconds=time.monotonic()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=="darwin" else 1024**2))
    dump(args.output/"results.json",payload)
    print(json.dumps(dict(event="complete",comparisons=comparisons)),flush=True)


if __name__=="__main__":
    main()
