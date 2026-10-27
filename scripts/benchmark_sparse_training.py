"""Validation-only historical support ablation with a frozen selected graph.

One 300-tree CatBoost per arm/seed. Historical features retain their own
prediction cutoffs; source tables, peer/history cohorts and operation plan
are unchanged. Controls must match the original full one-cutoff fit exactly.
"""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import time

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd

import kurversc
from kurversc.modeling import (
    fit_final_catboost,
    prepare_prediction_features,
    sample_training_rows,
    score_estimator,
)
from kurversc.training_support import (
    available_root_training,
    recency_weights,
    select_training_support,
)
from kurversc.staged_adapter import resolve_cache_config
from kurversc.task_source import load_task_source as RelBenchDatasetTask
from kurversc.staged_adapter import (
    KURVERSC_DEFAULTS,
    _SelectedGraph,
    _runtime_config,
)
from scripts.benchmark_frozen_event_summaries import (
    _capture,
    _checkpoint,
    _fold_tables,
    _prediction_score,
)
from scripts.run_kurversc_selected_relarena import AuditWinnerSystem
from kurversc.plan_dates import atomic_execution_plan
from scripts.temporal_campaign_support import required_policy_cutoffs
from scripts.frozen_materialization_cache import (
    FrozenMaterializationCache,
    assert_cache_equivalent,
)


IMPLEMENTATION_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
    + Path(kurversc.__file__).with_name("training_support.py").read_bytes()
).hexdigest()


def block_interval(truth, control, challenger, kind, groups):
    """Descriptive paired bootstrap by validation cutoff, not independent rows."""
    rng = np.random.default_rng(42)
    unique = np.unique(groups)
    if len(unique) < 2:
        return None
    blocks = [np.flatnonzero(groups == key) for key in unique]
    differences = []
    for _ in range(400):
        pos = np.concatenate(
            [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        )
        if kind == "classification" and np.unique(truth[pos]).size < 2:
            continue
        a = _prediction_score(truth[pos], control[pos], kind)
        b = _prediction_score(truth[pos], challenger[pos], kind)
        differences.append(b - a if kind == "classification" else a - b)
    return np.quantile(differences, [0.025, 0.975]).tolist() if differences else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", choices=("current", "prior"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-floors", default="500,1000")
    parser.add_argument("--frame-counts", default="3,12")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--atomic-date-replay", action="store_true")
    parser.add_argument("--policy-origins", type=int)
    parser.add_argument("--duckdb-memory-limit", default="6GB")
    parser.add_argument("--duckdb-temp-limit", default="80GB")
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Save verified historical frames without fitting ablation arms",
    )
    parser.add_argument(
        "--stream-feature-cache", action="store_true",
        help="Write unchanged full frame captures to disk instead of retaining every history copy",
    )
    args = parser.parse_args()
    if args.stream_feature_cache and not args.materialize_only:
        parser.error("--stream-feature-cache requires --materialize-only")
    if args.policy_origins is not None and (
        args.policy_origins < 2 or args.fold != "current" or not args.materialize_only
    ):
        parser.error(
            "--policy-origins requires current materialize-only and at least two origins"
        )
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint, trial = _checkpoint(args.dataset, args.task)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.atomic_date_replay:
        trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    source = RelBenchDatasetTask(args.dataset, args.task)
    inner = source.inner_split()
    task = source.task
    train_table, val_table = _fold_tables(inner, task, args.fold)
    schema = trial.execution_plan["kurversc_feature_schema"]
    columns = list(schema["columns"])
    categorical = set(schema["categorical_columns"])
    datetime = set(schema["datetime_columns"])
    kind = (
        "classification"
        if "classification" in str(task.task_type).lower()
        else "regression"
    )
    evaluation_cutoff = pd.to_datetime(val_table.df[task.time_col]).min()
    root = inner.db_state.table_dict[task.entity_table]
    support_training = available_root_training(
        train_table.df,
        root.df,
        label_key=task.entity_col,
        root_key=root.pkey_col,
        timestamp=task.time_col,
        root_timestamp=root.time_col,
    )
    plans = {
        "latest1": select_training_support(
            support_training,
            timestamp=task.time_col,
            evaluation_cutoff=evaluation_cutoff,
            label_horizon=task.timedelta,
            minimum_rows=1,
            max_frames=1,
        ),
        **{
            f"pooled{n}": select_training_support(
                support_training,
                timestamp=task.time_col,
                evaluation_cutoff=evaluation_cutoff,
                label_horizon=task.timedelta,
                minimum_rows=1,
                minimum_frames=min(n, args.max_frames),
                max_frames=args.max_frames,
            )
            for n in map(int, args.frame_counts.split(","))
        },
        **{
            f"floor{n}": select_training_support(
                support_training,
                timestamp=task.time_col,
                evaluation_cutoff=evaluation_cutoff,
                label_horizon=task.timedelta,
                minimum_rows=n,
                max_frames=args.max_frames,
            )
            for n in map(int, args.row_floors.split(","))
        },
    }
    all_cutoffs = tuple(
        sorted(set(cut for plan in plans.values() for cut in plan.cutoffs))
    )
    planned_folds = None
    if args.policy_origins is not None:
        required, planned_folds = required_policy_cutoffs(
            support_training,
            val_table.df,
            timestamp=task.time_col,
            horizon=task.timedelta,
            boundary_lag=evaluation_cutoff - pd.Timestamp(inner.cutoff),
            task=kind,
            origins=args.policy_origins,
        )
        # The original current training cohort is always retained for exact
        # overlapping-frame verification and the incumbent fallback.
        all_cutoffs = tuple(sorted(set(required) | set(plans["latest1"].cutoffs)))
    cache_file = args.output / "frames.json"
    KURVERSC_DEFAULTS.update(
        full_training_frames=1,
        full_validation_frames=None,
        base_distribution_features=True,
        temporal_estimator_mode="frame_ensemble",
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_max_temp_directory_size=args.duckdb_temp_limit,
    )
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        if cache.get("atomic_date_replay", False) != args.atomic_date_replay:
            raise ValueError("Cached frames use a different date replay implementation")
        if cache["checkpoint_sha256"] != digest or cache["cutoffs"] != list(
            map(str, all_cutoffs)
        ):
            raise ValueError(
                "Historical frame cache does not match the selected support"
            )
        headers = {"columns": ["__event_cutoff"]} if args.stream_feature_cache else {}
        train = pd.read_parquet(args.output / "history.parquet", **headers)
        val = pd.read_parquet(args.output / "validation.parquet", **headers)
        original_score = cache["control_score"]
    else:
        canonical = FrozenMaterializationCache(args.output, columns=columns,
            categorical=categorical, datetime=datetime)
        system = AuditWinnerSystem(
            cache=resolve_cache_config(None, on_miss="raise"),
            run_identity=source.run_identity(),
        )
        system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
        with canonical.install(), _capture(columns, task.target_col, task.entity_col, task.entity_table) as (
            baseline,
            _,
        ):
            result, original_problem = system._fit_arm(
                task,
                inner.db_state,
                train_table,
                val_table,
                seed=0,
                phase="outer",
                selection=_SelectedGraph(
                    config=trial.config,
                    execution_plan=copy.deepcopy(trial.execution_plan),
                ),
            )
        original_score = float(result.full_validation_score)
        baseline_train, baseline_val = (
            baseline[0],
            pd.concat(baseline[1:], ignore_index=True),
        )
        del result
        print(
            json.dumps(
                {
                    "phase": "control_verified_lifecycle",
                    "score": original_score,
                    "training_rows": len(baseline_train),
                    "validation_rows": len(baseline_val),
                    "historical_frames": len(all_cutoffs),
                    "features": len(columns),
                }
            ),
            flush=True,
        )
        # Preserve FULL original feature/history sources. Restrict only the
        # downstream label source, avoiding the adapter's stratified dates.
        # Reuse the exact original graph/provider blueprint as well as its
        # operation plan. Wider label support must not rebuild feature state.
        problem = original_problem
        labels = problem.label_node
        historical = train_table.df.loc[
            pd.to_datetime(train_table.df[task.time_col]).isin(all_cutoffs)
        ].copy()
        historical[labels.split] = labels.train_value
        validation_source = labels.source.loc[
            labels.source[labels.split] != labels.train_value
        ].copy()
        selected = pd.concat([historical, validation_source], ignore_index=True)
        # Changing label support must never change the authoritative unsampled
        # observed-label relation used to build the frozen graph's features.
        history_source = labels.history_source if labels.history_source is not None else labels.source
        problem = replace(problem, label_node=replace(labels, source=selected,
            history_source=history_source))
        config = _runtime_config()
        config.update(
            full_training_frames=len(all_cutoffs), temporal_estimator_mode="pooled"
        )
        options = dict(**problem.fit_kwargs(), **config, random_state=0,
            preselected_config=trial.config,
            preselected_execution_plan=copy.deepcopy(trial.execution_plan))
        if args.stream_feature_cache:
            from scripts.streaming_feature_cache import capture_to_disk
            with canonical.install(replay=True), capture_to_disk(columns, task.target_col, task.entity_col,
                    task.entity_table, args.output / "captured-frames") as sink:
                pooled = kurversc.fit(**options)
            sink.concatenate(args.output / "history.parquet", cutoffs=all_cutoffs)
            validation_cuts = [r["cutoff"] for r in sink.parts if pd.Timestamp(r["cutoff"]) not in all_cutoffs]
            sink.concatenate(args.output / "validation.parquet", cutoffs=validation_cuts)
            latest = pd.read_parquet(args.output / "history.parquet",
                filters=[("__event_cutoff", "==", str(all_cutoffs[-1]))])
            train = pd.read_parquet(args.output / "history.parquet", columns=["__event_cutoff"])
            val = pd.read_parquet(args.output / "validation.parquet")
        else:
            with canonical.install(replay=True), _capture(columns, task.target_col, task.entity_col, task.entity_table) as (frames, cuts):
                pooled = kurversc.fit(**options)
            training_frames = [frame for frame, cutoff in zip(frames, cuts) if pd.Timestamp(cutoff) in all_cutoffs]
            validation_frames = [frame for frame, cutoff in zip(frames, cuts) if pd.Timestamp(cutoff) not in all_cutoffs]
            train, val = pd.concat(training_frames, ignore_index=True), pd.concat(validation_frames, ignore_index=True)
            latest = train.loc[train["__event_cutoff"] == str(all_cutoffs[-1])].reset_index(drop=True)
        # Both fits consume the SAME immutable overlapping cohorts. Do not
        # recompute floating aggregates and then accept a numerical tolerance.
        # Parquet's None/pd.NA and list/ndarray storage differences are checked
        # canonically; all numeric and prepared learner values remain exact.
        for actual, reference in ((latest, baseline_train), (val, baseline_val)):
            assert_cache_equivalent(actual, reference, columns=columns,
                categorical=categorical, datetime=datetime)
        if canonical.hits != len(baseline):
            raise ValueError("Every original control cohort must be reused exactly once")
        canonical_report = dict(strategy="single_frozen_materialization",
            replayed_control_cohorts=canonical.hits, independent_overlap_recomputations=0,
            numeric_tolerance_used=False, model_visible_values_identical=True)
        canonical.close()
        if not args.stream_feature_cache:
            train.to_parquet(args.output / "history.parquet", index=False)
            val.to_parquet(args.output / "validation.parquet", index=False)
        cache = {
            "atomic_date_replay": args.atomic_date_replay,
            "checkpoint_sha256": digest,
            "cutoffs": list(map(str, all_cutoffs)),
            "control_score": original_score,
            "frozen_plan_fingerprint": pooled.plan_fingerprint,
            "overlapping_frames_identical": True,
            "streamed_feature_capture": args.stream_feature_cache,
            "canonical_overlap": canonical_report,
        }
        cache_file.write_text(json.dumps(cache, indent=2) + "\n")
        del pooled
        if args.stream_feature_cache:
            # Preserve only alignment headers after the full incumbent-frame
            # and prepared-model-value equivalence checks above have passed.
            val = val[["__event_cutoff"]]
            del baseline, baseline_train, baseline_val, latest
    if (
        pd.to_datetime(train["__event_cutoff"]) + task.timedelta > evaluation_cutoff
    ).any():
        raise RuntimeError("Immature labels reached downstream training")
    if args.materialize_only:
        metadata = dict(
            dataset=args.dataset,
            task=args.task,
            fold=args.fold,
            atomic_date_replay=args.atomic_date_replay,
            checkpoint_sha256=digest,
            checkpoint=str(checkpoint),
            feature_schema=schema,
            target=task.target_col,
            **({"entity_table":task.entity_table,"entity_col":task.entity_col,
                "label_available_at":task.label_available_at,
                "phase_boundary_lag":str(evaluation_cutoff-pd.Timestamp(inner.cutoff))}
               if getattr(task,"label_available_at",None) else {}),
            label_horizon=str(task.timedelta),
            metric="roc_auc" if kind == "classification" else "mae",
            validation_cutoffs=int(val["__event_cutoff"].nunique()),
            history_rows=len(train),
            validation_rows=len(val),
            control_score=float(original_score),
            retrospective_folds=[list(map(str, fold)) for fold in planned_folds]
            if planned_folds is not None
            else None,
            scope="Verified frozen inner-split feature cache; no official test",
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
            streamed_feature_capture=args.stream_feature_cache,
        )
        (args.output / "materialization.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps(dict(phase="materialized", **metadata)), flush=True)
        return
    x, vx = [
        prepare_prediction_features(
            frame,
            columns=columns,
            categorical_columns=categorical,
            datetime_columns=datetime,
        )
        for frame in (train, val)
    ]
    cat_indices = [i for i, col in enumerate(columns) if col in categorical]
    truth = pd.to_numeric(val[task.target_col]).to_numpy(dtype=float)
    records, predictions = [], {}
    largest = max(plans, key=lambda name: len(plans[name].cutoffs))
    arms = [*plans, "recency_support"]
    for arm in arms:
        plan = plans[largest if arm == "recency_support" else arm]
        mask = pd.to_datetime(train["__event_cutoff"]).isin(plan.cutoffs)
        tx, ty = (
            x.loc[mask].reset_index(drop=True),
            train.loc[mask, task.target_col].reset_index(drop=True),
        )
        times = pd.to_datetime(train.loc[mask, "__event_cutoff"]).reset_index(drop=True)
        arm_predictions = []
        for seed in map(int, args.seeds.split(",")):
            fit_started = time.perf_counter()
            # Preserve row indices so the same deterministic sample applies
            # to recency weights as to features/labels on denser transfers.
            sx, sy = sample_training_rows(
                tx, ty, limit=100_000, task=kind, random_state=seed + 1
            )
            if arm == "recency_support":
                half_life = max(
                    1.0, (times.max() - times.min()).total_seconds() / 86_400 / 2
                )
                weights = recency_weights(times, half_life_days=half_life)
                params = dict(
                    iterations=300,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=10.0,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=8,
                )
                model = (
                    CatBoostClassifier(
                        loss_function="Logloss", eval_metric="AUC", **params
                    )
                    if kind == "classification"
                    else CatBoostRegressor(
                        loss_function="MAE", eval_metric="MAE", **params
                    )
                )
                model.fit(
                    sx,
                    sy,
                    cat_features=cat_indices,
                    sample_weight=weights.loc[sx.index],
                )
                classes = tuple(sorted(ty.unique())) if kind == "classification" else ()
            else:
                model, _, classes = fit_final_catboost(
                    sx,
                    sy,
                    task=kind,
                    categorical=cat_indices,
                    random_state=seed,
                    model_params=None,
                )
            metric, score, prediction = score_estimator(
                model, vx, val[task.target_col], task=kind, target_classes=classes
            )
            if arm == "latest1" and seed == 0 and abs(score - original_score) > 1e-9:
                raise RuntimeError(
                    f"Frozen single-cutoff control mismatch: {score} != {original_score}"
                )
            record = dict(
                arm=arm,
                seed=seed,
                score=score,
                metric=metric,
                training_rows=len(tx),
                estimator_rows=len(sx),
                training_frames=len(plan.cutoffs),
                first_training_cutoff=str(plan.cutoffs[0]),
                last_training_cutoff=str(plan.cutoffs[-1]),
                distinct_roots=int(train.loc[mask, "__event_entity"].nunique()),
                model_seconds=time.perf_counter() - fit_started,
            )
            records.append(record)
            np.savez_compressed(
                args.output / f"{arm}-seed{seed}.npz",
                truth=truth,
                prediction=prediction,
            )
            arm_predictions.append(prediction)
            print(json.dumps(record), flush=True)
            del model
        predictions[arm] = np.mean(arm_predictions, axis=0)
    control = _prediction_score(truth, predictions["latest1"], kind)
    comparison = {}
    for arm, pred in predictions.items():
        score = _prediction_score(truth, pred, kind)
        comparison[arm] = dict(
            score=score,
            improvement=score - control
            if kind == "classification"
            else control - score,
            block_bootstrap_95pct=block_interval(
                truth,
                predictions["latest1"],
                pred,
                kind,
                val["__event_cutoff"].to_numpy(),
            ),
        )
    payload = dict(
        atomic_date_replay=args.atomic_date_replay,
        dataset=args.dataset,
        task=args.task,
        fold=args.fold,
        metric=metric,
        scope="validation only; selected graph frozen; no official test",
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest,
        implementation_sha256=IMPLEMENTATION_SHA256,
        feature_count=len(columns),
        validation_rows=len(val),
        validation_cutoffs=int(val["__event_cutoff"].nunique()),
        raw_training_label_rows=len(train_table.df),
        root_available_training_label_rows=len(support_training),
        plans={
            name: {**asdict(plan), "cutoffs": list(map(str, plan.cutoffs))}
            for name, plan in plans.items()
        },
        records=records,
        comparison=comparison,
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        versions={
            name: version(name) for name in ("pandas", "duckdb", "catboost", "relbench")
        },
        caveat="Graph selected on current validation; earlier period is a retrospective stability check, not an independent graph-selection holdout. Seed-average predictions are diagnostic only; production remains single seed.",
    )
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": "complete",
                "comparison": comparison,
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
