#!/usr/bin/env python3
"""Compare regression heads on a frozen current KurveRSC graph, validation only.

RelArena's inner database is used; no official test predictions are requested.
The winning GraphReduce plan and all original CatBoost inputs are unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import pickle
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import kurversc.core as core
from kurversc.modeling import prepare_prediction_features
from kurversc.ordered_regression import OrderedTargetRegressor
from kurversc.staged_adapter import resolve_cache_config, KURVERSC_DEFAULTS, _SelectedGraph
from kurversc.task_source import load_task_source as RelBenchDatasetTask
from scripts.run_kurversc_selected_relarena import AuditWinnerSystem


def _fold_tables(inner, task, fold: str):
    if fold == "current":
        return inner.train_table, inner.eval_table
    source = inner.train_table
    cutoff = source.df[task.time_col].max()
    train = copy.copy(source)
    validation = copy.copy(source)
    train.df = source.df.loc[source.df[task.time_col] < cutoff].copy()
    validation.df = source.df.loc[source.df[task.time_col] == cutoff].copy()
    if train.df.empty or validation.df.empty:
        raise ValueError("Prior chronological fold is empty")
    return train, validation


@contextmanager
def _capture_frozen_frames(columns: tuple[str, ...], target: str):
    original = core._materialize_at_cutoff
    frames: list[pd.DataFrame] = []

    def materialize(*args, **kwargs):
        frame = original(*args, **kwargs)
        missing = set((*columns, target)) - set(frame.columns)
        if missing:
            raise RuntimeError(f"Frozen frame missing selected columns: {sorted(missing)[:8]}")
        frames.append(frame.loc[:, [*columns, target]].copy())
        return frame

    core._materialize_at_cutoff = materialize
    try:
        yield frames
    finally:
        core._materialize_at_cutoff = original


def _save_prediction(output: Path, name: str, truth, prediction) -> float:
    estimate = np.asarray(prediction, dtype=float).reshape(-1)
    actual = np.asarray(truth, dtype=float).reshape(-1)
    if len(actual) != len(estimate):
        raise ValueError("Prediction and target lengths differ")
    np.savez_compressed(output / f"{name}-validation.npz", truth=actual, prediction=estimate)
    return float(mean_absolute_error(actual, estimate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-trial")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", choices=("prior", "current"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duckdb-memory-limit", default="12GB")
    args = parser.parse_args()
    with args.checkpoint.open("rb") as handle:
        trial = pickle.load(handle)["trial"]  # noqa: S301 - trusted local checkpoint
    if trial.stage != "rerank" or trial.status != "completed" or not trial.execution_plan:
        raise ValueError("Expected completed rerank checkpoint with frozen plan")

    started = time.perf_counter()
    source = RelBenchDatasetTask(args.dataset, args.task)
    if "regression" not in str(source.task.task_type).lower():
        raise ValueError("This readout ablation requires a regression task")
    inner = source.inner_split()
    train_table, val_table = _fold_tables(inner, source.task, args.fold)
    KURVERSC_DEFAULTS.update(
        full_training_frames=1,
        base_distribution_features=True,
        temporal_estimator_mode="frame_ensemble",
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_max_temp_directory_size="100GB",
    )
    system = AuditWinnerSystem(
        cache=resolve_cache_config(None, on_miss="raise"),
        run_identity=source.run_identity(),
    )
    system.peer_context_mode = "both"
    system.peer_max_history_rows = 50_000
    selection = _SelectedGraph(
        config=trial.config,
        execution_plan=copy.deepcopy(trial.execution_plan),
    )
    schema = trial.execution_plan["kurversc_feature_schema"]
    columns = tuple(schema["columns"])
    target = source.task.target_col
    args.output.mkdir(parents=True, exist_ok=True)
    with _capture_frozen_frames(columns, target) as frames:
        result, _ = system._fit_arm(
            source.task,
            inner.db_state,
            train_table,
            val_table,
            seed=args.seed,
            phase="outer",
            selection=selection,
        )
    if len(frames) != 2:
        raise RuntimeError(f"Expected one train and one validation frame, got {len(frames)}")
    train_frame, val_frame = frames
    train_frame.to_parquet(args.output / "train-frame.parquet", index=False)
    val_frame.to_parquet(args.output / "validation-frame.parquet", index=False)
    categorical = tuple(schema["categorical_columns"])
    datetime = tuple(schema["datetime_columns"])
    x = prepare_prediction_features(
        train_frame,
        columns=list(columns),
        categorical_columns=set(categorical),
        datetime_columns=set(datetime),
    )
    vx = prepare_prediction_features(
        val_frame,
        columns=list(columns),
        categorical_columns=set(categorical),
        datetime_columns=set(datetime),
    )
    y = pd.to_numeric(train_frame[target]).astype(float)
    vy = pd.to_numeric(val_frame[target]).astype(float)
    control = _save_prediction(
        args.output,
        "control",
        vy,
        result.fitted_model.validation_estimator.predict(vx),
    )
    if abs(control - float(result.full_validation_score)) > 1e-9:
        raise RuntimeError("Captured control frame does not reproduce KurveRSC MAE")
    scores = {"control_mae": control}
    print(json.dumps({"head": "control", "mae": control, "features": len(columns)}), flush=True)

    model = OrderedTargetRegressor(random_state=args.seed, thread_count=8)
    model.fit(x, y, cat_features=list(categorical))
    scores["ordered_median_mae"] = _save_prediction(
        args.output, "ordered_median", vy, model.predict(vx, statistic="median")
    )
    scores["ordered_mean_mae"] = _save_prediction(
        args.output, "ordered_mean", vy, model.predict(vx, statistic="mean")
    )
    scores["ordered_fit_audit"] = model.fit_audit_
    print(json.dumps({"head": "ordered_median", "mae": scores["ordered_median_mae"]}), flush=True)

    if (args.dataset, args.task) == ("rel-trial", "study-adverse"):
        huber = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=10.0,
            loss_function="Huber:delta=50",
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=8,
        )
        huber.fit(x, y, cat_features=list(categorical))
        scores["huber_mae"] = _save_prediction(
            args.output, "huber", vy, huber.predict(vx)
        )
        print(json.dumps({"head": "huber", "mae": scores["huber_mae"]}), flush=True)

    payload = {
        "task": f"{args.dataset}/{args.task}",
        "fold": args.fold,
        "seed": args.seed,
        "validation_cutoff": str(val_table.df[source.task.time_col].iloc[0]),
        "training_rows": len(train_frame),
        "validation_rows": len(val_frame),
        "feature_count": len(columns),
        "plan_fingerprint": result.plan_fingerprint,
        "checkpoint": str(args.checkpoint),
        "scores": scores,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "frozen selected graph, RelArena inner database, no official test",
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
