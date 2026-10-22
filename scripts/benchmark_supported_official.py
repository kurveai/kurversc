"""Official-test diagnostic of a policy frozen from validation evidence.

Graph/operations/providers remain frozen. Only historical fitting support and
the scalar versus ordered-distribution readout change. All test predictions
are saved before the official scorer is invoked. No test-based selection.
"""

import argparse
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import re
import resource
import time

import numpy as np
import pandas as pd

import kurversc
import kurversc.core as core
from kurversc.modeling import (
    fit_final_catboost,
    prepare_prediction_features,
    sample_training_rows,
)
from kurversc.ordered_regression import OrderedTargetRegressor
from kurversc.plan_dates import atomic_date_replay, atomic_execution_plan
from kurversc.training_support import available_root_training, select_training_support
from kurversc.staged_adapter import resolve_cache_config
from kurversc.task_source import load_task_source as RelBenchDatasetTask
from kurversc.staged_adapter import (
    KURVERSC_DEFAULTS,
    _SelectedGraph,
    _runtime_config,
)
from scripts.benchmark_frozen_event_summaries import _capture, _checkpoint
from scripts.run_kurversc_selected_relarena import AuditWinnerSystem


@contextmanager
def ordered_readout(enabled):
    original = core._fit_temporal_frame_estimator

    def fit(
        features,
        labels,
        *,
        model_backend,
        task,
        categorical,
        random_state,
        model_params,
        target_classes=(),
        iterations=None,
    ):
        if model_backend != "catboost" or task != "regression":
            raise ValueError(
                "Ordered readout experiment requires CatBoost numeric regression"
            )
        started = time.perf_counter()
        model = OrderedTargetRegressor(
            iterations=iterations or 300,
            random_state=random_state,
            thread_count=4,
            l2_leaf_reg=10.0,
        ).fit(features, labels, cat_features=categorical)
        return model, time.perf_counter() - started, ()

    if enabled:
        core._fit_temporal_frame_estimator = fit
    try:
        yield
    finally:
        core._fit_temporal_frame_estimator = original


def support_budget(evidence, arm):
    """Interpret an explicit validation-tested policy, not its observed row count."""
    if "selected_policy" in evidence:
        spec = evidence["selected_policy"]
        if arm != spec["name"] or spec["mode"] != "pooled":
            raise ValueError(
                "Only the validation-selected pooled challenger may be replayed"
            )
        return (
            int(spec["minimum_training_rows"]),
            int(spec["training_frames"]),
            int(spec["max_training_frames"]),
            spec["readout"] == "ordered_median",
        )
    if "plans" in evidence:
        if arm not in evidence["plans"]:
            raise ValueError(
                "Only unweighted direct historical-support arms are supported"
            )
        spec = evidence["plans"][arm]
        if arm.startswith("floor"):
            return int(spec["minimum_rows"]), 1, 64, False
        match = re.fullmatch(r"(?:latest|pooled)(\d+)", arm)
        if not match:
            raise ValueError("Unrecognized training support policy")
        frames = int(match[1])
        return 1, frames, max(frames, 1), False
    if arm not in {"control", "ordered_median"}:
        raise ValueError("Only direct or ordered-median readout is supported")
    match = re.match(r"(?:latest|pooled)(\d+)", evidence["support"])
    if not match:
        raise ValueError("Evidence does not declare a reproducible support policy")
    frames = int(match[1])
    return 1, frames, frames, arm == "ordered_median"


def selected_policy_evidence(path):
    """Read a completed validation-only selector; never select from test scores."""
    audit = json.loads(path.read_text())
    manifest = json.loads((path.parent / "manifest.json").read_text())
    selected = audit["selected_policy"]
    if audit["decision"]["selected"] != selected["name"]:
        raise ValueError("Recorded selected policy differs from the audited decision")
    if (
        selected["name"] != "incumbent"
        and audit["decision"]["candidates"][selected["name"]]["reason"] != "eligible"
    ):
        raise ValueError("Challenger was not admitted by the validation gate")
    metadata = manifest["metadata"]
    if (audit["dataset"], audit["task"]) != (metadata["dataset"], metadata["task"]):
        raise ValueError("Selector task differs from its immutable manifest")
    if (
        selected["name"] != "incumbent"
        and metadata["validation_cutoffs"] > 1
        and not manifest.get("phase_frozen_cache")
    ):
        raise ValueError("Multi-cutoff selector lacks phase-frozen validation features")
    records = [{**r, "arm": r["policy"]} for r in audit["records"] if "score" in r]
    latest = max(pd.Timestamp(r["origin"]) for r in records)
    records = [r for r in records if pd.Timestamp(r["origin"]) == latest]
    return dict(
        dataset=audit["dataset"],
        task=audit["task"],
        fold="current",
        checkpoint_sha256=metadata["checkpoint_sha256"],
        selected_policy=selected,
        records=records,
        selection_audit=audit["decision"],
    ), selected["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--evidence", type=Path)
    source_group.add_argument("--policy-audit", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reselect-final-support", action="store_true")
    parser.add_argument("--duckdb-memory-limit", default="6GB")
    parser.add_argument("--duckdb-temp-limit", default="80GB")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.policy_audit:
        if args.arm is not None or args.reselect_final_support:
            parser.error(
                "A selector audit fixes the arm and its final support lifecycle"
            )
        evidence, args.arm = selected_policy_evidence(args.policy_audit)
        args.evidence = args.policy_audit
    else:
        if args.arm is None:
            parser.error("Manual validation evidence requires --arm")
        evidence = json.loads(args.evidence.read_text())
    if (evidence["dataset"], evidence["task"], evidence["fold"]) != (
        args.dataset,
        args.task,
        "current",
    ):
        raise ValueError("Evidence must be completed current validation for this task")
    record = next(
        r for r in evidence["records"] if r["arm"] == args.arm and r["seed"] == 0
    )
    checkpoint, trial = _checkpoint(args.dataset, args.task)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != evidence["checkpoint_sha256"]:
        raise ValueError("Selected graph differs from the validation-tested graph")
    trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    incumbent_selected = args.policy_audit is not None and args.arm == "incumbent"
    if incumbent_selected:
        row_floor, minimum_frames, maximum_frames, ordered = 1, 1, 1, False
    else:
        row_floor, minimum_frames, maximum_frames, ordered = support_budget(
            evidence, args.arm
        )
    manifest = dict(
        dataset=args.dataset,
        task=args.task,
        arm=args.arm,
        validation_seed0_score=record["score"],
        evidence=str(args.evidence),
        evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest,
        minimum_rows=row_floor,
        minimum_frames=minimum_frames,
        maximum_frames=maximum_frames,
        head="ordered_median" if ordered else "direct",
        temporal_mode="frame_ensemble" if incumbent_selected else "pooled",
        seed=0,
        tree_budget=300,
        atomic_date_replay=True,
        reselect_final_support=args.reselect_final_support,
        implementation_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(core.__file__).with_name("plan_dates.py").read_bytes()
        ).hexdigest(),
        scope="Explicit policy frozen before outer split/test scoring. Same graph and feature providers. The frame-ensemble incumbent includes all observed validation-frame refits; a pooled challenger includes those observed frames in its pooled refit. Optional final support reselection reapplies the same target-blind rule to known materialized labels at the test forecast origin. This mode was motivated by earlier test diagnostics and is not independent of that exploratory feedback.",
        validation_selector=bool(args.policy_audit),
        incumbent_selected=incumbent_selected,
        blend_weight=float(
            evidence.get("selected_policy", {}).get("blend_weight", 1.0)
        ),
    )
    if not 0 < manifest["blend_weight"] <= 1:
        raise ValueError("Validation-selected blend weight must be in (0, 1]")
    manifest_path = args.output / "policy.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing official experiment has a different frozen policy")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Freeze every policy choice above before exposing the outer split.
    source = RelBenchDatasetTask(args.dataset, args.task)
    outer = source.outer_split()
    task = source.task
    from kurversc.prediction_provenance import masked_query_keys, key_fingerprint, key_arrays
    prediction_keys = masked_query_keys(outer.eval_table.df, entity=task.entity_col,
                                       timestamp=task.time_col, target=task.target_col)
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_keys.to_parquet(args.output / "masked-query-keys.parquet", index=False)
    KURVERSC_DEFAULTS.update(
        full_training_frames=1,
        full_validation_frames=None,
        base_distribution_features=True,
        temporal_estimator_mode="frame_ensemble",
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_max_temp_directory_size=args.duckdb_temp_limit,
    )
    system = AuditWinnerSystem(
        cache=resolve_cache_config(None, on_miss="raise"),
        run_identity=source.run_identity(),
    )
    system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
    # Serialized per-graph replay now works without the prototype's class patch.
    with atomic_date_replay(enabled=False):
        result, problem = system._fit_arm(
            task,
            outer.db_state,
            outer.train_table,
            outer.val_table,
            seed=0,
            phase="outer",
            selection=_SelectedGraph(
                config=trial.config, execution_plan=copy.deepcopy(trial.execution_plan)
            ),
        )
        control_predictions = system._predict(
            task, outer.eval_table, result=result, problem=problem
        )
        control_validation_score = result.full_validation_score
        if incumbent_selected:
            np.savez_compressed(
                args.output / "predictions.npz",
                control=control_predictions,
                candidate=control_predictions.copy(),
                **key_arrays(prediction_keys),
            )
            score = float(
                task.evaluate(control_predictions, None, metrics=[source.metric])[
                    source.metric.__name__
                ]
            )
            payload = dict(
                **manifest,
                metric=source.metric.__name__,
                official_test_scores=dict(control=score, candidate=score),
                control_outer_validation_score=control_validation_score,
                candidate_outer_validation_score=control_validation_score,
                elapsed_seconds=time.perf_counter() - started,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**3,
                policy_applied="Exact incumbent retained; no challenger fit or test-based selection",
                query_keys_sha256=key_fingerprint(prediction_keys),
            )
            (args.output / "results.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        phase="official_complete",
                        scores=payload["official_test_scores"],
                    )
                ),
                flush=True,
            )
            return
        del result
        gc.collect()
        root = outer.db_state.table_dict[task.entity_table]
        available = available_root_training(
            outer.train_table.df,
            root.df,
            label_key=task.entity_col,
            root_key=root.pkey_col,
            timestamp=task.time_col,
            root_timestamp=root.time_col,
        )
        evaluation_cutoff = pd.to_datetime(outer.val_table.df[task.time_col]).min()
        support = select_training_support(
            available,
            timestamp=task.time_col,
            evaluation_cutoff=evaluation_cutoff,
            label_horizon=task.timedelta,
            minimum_rows=row_floor,
            minimum_frames=minimum_frames,
            max_frames=maximum_frames,
        )
        labels = problem.label_node
        historical = outer.train_table.df.loc[
            pd.to_datetime(outer.train_table.df[task.time_col]).isin(support.cutoffs)
        ].copy()
        historical[labels.split] = labels.train_value
        validation = labels.source.loc[
            labels.source[labels.split] != labels.train_value
        ].copy()
        observed = pd.concat([historical, validation], ignore_index=True)
        first_test_cutoff = pd.to_datetime(outer.eval_table.df[task.time_col]).min()
        if (
            pd.to_datetime(observed[task.time_col]) + task.timedelta > first_test_cutoff
        ).any():
            raise RuntimeError("Immature labels would enter official-test fitting")
        problem = replace(problem, label_node=replace(labels, source=observed))
        config = _runtime_config()
        config.update(
            full_training_frames=len(support.cutoffs), temporal_estimator_mode="pooled"
        )
        print(
            json.dumps(
                dict(
                    phase="supported_outer_started",
                    support_cutoffs=list(map(str, support.cutoffs)),
                    historical_rows=len(historical),
                    observed_validation_rows=len(validation),
                    head=manifest["head"],
                )
            ),
            flush=True,
        )
        schema = trial.execution_plan["kurversc_feature_schema"]
        capture = (
            _capture(
                schema["columns"], task.target_col, task.entity_col, task.entity_table
            )
            if args.reselect_final_support
            else nullcontext(([], []))
        )
        final_support = None
        final_training_rows = None
        with ordered_readout(ordered), capture as (frames, cutoffs):
            candidate = kurversc.fit(
                **problem.fit_kwargs(),
                **config,
                random_state=0,
                preselected_config=trial.config,
                preselected_execution_plan=copy.deepcopy(trial.execution_plan),
            )
        if args.reselect_final_support:
            materialized = pd.concat(frames, ignore_index=True)
            final_support = select_training_support(
                materialized,
                timestamp="__event_cutoff",
                evaluation_cutoff=first_test_cutoff,
                label_horizon=task.timedelta,
                minimum_rows=row_floor,
                minimum_frames=minimum_frames,
                max_frames=maximum_frames,
            )
            chosen = materialized.loc[
                pd.to_datetime(materialized.__event_cutoff).isin(final_support.cutoffs)
            ]
            x = prepare_prediction_features(
                chosen,
                columns=list(schema["columns"]),
                categorical_columns=set(schema["categorical_columns"]),
                datetime_columns=set(schema["datetime_columns"]),
            )
            tx, ty = sample_training_rows(
                x,
                chosen[task.target_col],
                limit=100_000,
                task=candidate.task,
                random_state=1,
            )
            cat_indices = [
                i
                for i, name in enumerate(schema["columns"])
                if name in set(schema["categorical_columns"])
            ]
            if ordered:
                model = OrderedTargetRegressor(
                    random_state=0, thread_count=4, l2_leaf_reg=10.0
                ).fit(tx, ty, cat_features=cat_indices)
                classes = ()
            else:
                model, _, classes = fit_final_catboost(
                    tx,
                    ty,
                    task=candidate.task,
                    categorical=cat_indices,
                    random_state=0,
                    model_params=None,
                )
            candidate.fitted_model.estimator = model
            candidate.fitted_model.target_classes = (
                classes or candidate.fitted_model.target_classes
            )
            final_training_rows = len(tx)
            print(
                json.dumps(
                    dict(
                        phase="final_support_reselected",
                        rows=final_training_rows,
                        cutoffs=list(map(str, final_support.cutoffs)),
                    )
                ),
                flush=True,
            )
            del frames, materialized, chosen, x, tx, ty
            gc.collect()
        candidate_predictions = system._predict(
            task, outer.eval_table, result=candidate, problem=problem
        )
        if manifest["blend_weight"] < 1:
            candidate_predictions = (
                1 - manifest["blend_weight"]
            ) * control_predictions + manifest["blend_weight"] * candidate_predictions
        if (
            not np.isfinite(control_predictions).all()
            or not np.isfinite(candidate_predictions).all()
        ):
            raise RuntimeError("Non-finite official predictions")
        np.savez_compressed(
            args.output / "predictions.npz",
            control=control_predictions,
            candidate=candidate_predictions,
            **key_arrays(prediction_keys),
        )
        # Both arms' predictions are fixed BEFORE any test score is requested.
        scores = {
            name: float(
                task.evaluate(prediction, None, metrics=[source.metric])[
                    source.metric.__name__
                ]
            )
            for name, prediction in (
                ("control", control_predictions),
                ("candidate", candidate_predictions),
            )
        }
        result = dict(
            **manifest,
            metric=source.metric.__name__,
            official_test_scores=scores,
            query_keys_sha256=key_fingerprint(prediction_keys),
            control_outer_validation_score=control_validation_score,
            candidate_outer_validation_score=(
                candidate.full_validation_score
                if manifest["blend_weight"] == 1
                else None
            ),
            pure_challenger_outer_validation_score=candidate.full_validation_score,
            selected_training_cutoffs=list(map(str, support.cutoffs)),
            support_rows=support.rows,
            final_training_rows=final_training_rows,
            reselected_final_cutoffs=list(map(str, final_support.cutoffs))
            if final_support is not None
            else None,
            feature_count=len(candidate.fitted_model.feature_columns),
            plan_fingerprint=candidate.plan_fingerprint,
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        )
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(phase="official_complete", scores=scores)), flush=True)


if __name__ == "__main__":
    main()
