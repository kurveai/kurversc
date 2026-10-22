#!/usr/bin/env python3
"""Matched validation ablations on a frozen current graph, never official test.

PYTHONPATH=../../graphreduce:.:.relarena-v2/src ../relarena/.venv/bin/python \
    -m scripts.benchmark_frozen_event_summaries --dataset rel-avito \
    --task ad-ctr --fold current --modes subday,conditional,subday+conditional \
    --output results/frozen-event-summaries-20261005/ad-ctr-current
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import resource
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

import kurversc.core as core
from kurversc.event_summaries import EventSummaryPlan, EventTable
from kurversc.modeling import (
    fit_final_catboost,
    prepare_prediction_features,
    sample_training_rows,
    score_estimator,
)
from kurversc.staged_adapter import resolve_cache_config
from kurversc.task_source import load_task_source as RelBenchDatasetTask
from kurversc.staged_adapter import KURVERSC_DEFAULTS, _SelectedGraph
from scripts.benchmark_current_frozen_heads import _fold_tables
from scripts.run_kurversc_selected_relarena import AuditWinnerSystem


CAMPAIGN = Path("results/kurversc-root-reachable-context-fixed-all21-20261003")
IMPLEMENTATION_SHA256 = hashlib.sha256(
    Path(core.__file__).with_name("event_summaries.py").read_bytes()
).hexdigest()


def _checkpoint(dataset, task):
    wanted = json.loads(
        json.loads((CAMPAIGN / "results" / f"{dataset}_{task}.json").read_text())[
            "config"
        ]
    )
    matches = []
    base = (
        CAMPAIGN
        / "checkpoints"
        / "staged-retention-base-cdf-on-peer-both-h50000"
        / f"{dataset}_{task}"
    )
    for path in sorted(base.glob("*/trials/*.pkl")):
        with path.open("rb") as handle:
            trial = pickle.load(handle)["trial"]  # noqa: S301 - trusted local artifacts
        config = {
            **asdict(trial.config),
            "feature_families": list(trial.config.feature_families),
        }
        if config == wanted and trial.status == "completed" and trial.execution_plan:
            matches.append((trial.stage != "rerank", str(path), path, trial))
    if not matches:
        raise ValueError(
            "No completed checkpoint matches the current selected configuration"
        )
    _, _, path, trial = min(matches, key=lambda item: item[:2])
    return path, trial


@contextmanager
def _capture(columns, target, entity, root):
    original = core._materialize_at_cutoff
    frames, cutoffs = [], []

    def materialize(*args, **kwargs):
        frame = original(*args, **kwargs)
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


def _prediction_score(truth, prediction, task):
    return float(
        roc_auc_score(truth, prediction)
        if task == "classification"
        else mean_absolute_error(truth, prediction)
    )


def _paired_interval(truth, control, challenger, task, repeats=400):
    # Descriptive paired row-bootstrap, not a guarantee across temporal shifts.
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(repeats):
        positions = rng.integers(0, len(truth), len(truth))
        if task == "classification" and np.unique(truth[positions]).size < 2:
            continue
        a = _prediction_score(truth[positions], control[positions], task)
        b = _prediction_score(truth[positions], challenger[positions], task)
        differences.append(b - a if task == "classification" else a - b)
    return np.quantile(differences, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", choices=("prior", "current"), required=True)
    parser.add_argument("--modes", default="subday")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame-cache",
        type=Path,
        help="Reuse a previously verified frozen-frame cache",
    )
    parser.add_argument("--max-numeric", type=int, default=4)
    parser.add_argument("--max-numeric-pairs", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=8)
    parser.add_argument(
        "--numeric-scope", choices=("both", "raw", "dimensions"), default="both"
    )
    parser.add_argument("--numeric-history-days", type=int, default=730)
    parser.add_argument("--duckdb-memory-limit", default="6GB")
    args = parser.parse_args()
    modes = args.modes.split(",")
    atoms = sorted(set(atom for mode in modes for atom in mode.split("+")))
    if set(atoms) - {"subday", "clock", "numeric", "conditional", "joint_numeric"}:
        parser.error("Unsupported event mode")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if seeds[0] != 0:
        parser.error(
            "The first seed must be zero for exact frozen-control verification"
        )
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    path, trial = _checkpoint(args.dataset, args.task)
    print(
        json.dumps(
            {"phase": "load", "checkpoint": str(path), "config": asdict(trial.config)}
        ),
        flush=True,
    )
    source = RelBenchDatasetTask(args.dataset, args.task)
    inner = source.inner_split()
    train_table, val_table = _fold_tables(inner, source.task, args.fold)
    schema = trial.execution_plan["kurversc_feature_schema"]
    columns, categorical = list(schema["columns"]), set(schema["categorical_columns"])
    datetime = set(schema["datetime_columns"])
    target = source.task.target_col
    task = (
        "classification"
        if "classification" in str(source.task.task_type).lower()
        else "regression"
    )
    cache_root = args.frame_cache or args.output
    cache_file = cache_root / "frozen-cache.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        if cache["checkpoint_sha256"] != digest or cache["fold"] != args.fold:
            raise ValueError("Frozen-frame cache does not match this experiment")
        frames = [
            pd.read_parquet(cache_root / f"{name}-frame.parquet")
            for name in ("train", "validation")
        ]
        cutoffs = cache["cutoffs"]
        original_score = cache["control_score"]
    else:
        if args.frame_cache:
            raise ValueError("Requested frozen-frame cache does not exist")
        KURVERSC_DEFAULTS.update(
            full_training_frames=1,
            base_distribution_features=True,
            full_validation_frames=None,
            temporal_estimator_mode="frame_ensemble",
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_max_temp_directory_size="80GB",
        )
        system = AuditWinnerSystem(
            cache=resolve_cache_config(None, on_miss="raise"),
            run_identity=source.run_identity(),
        )
        system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
        selection = _SelectedGraph(
            config=trial.config, execution_plan=copy.deepcopy(trial.execution_plan)
        )
        with _capture(
            columns, target, source.task.entity_col, source.task.entity_table
        ) as (frames, cutoffs):
            result, _ = system._fit_arm(
                source.task,
                inner.db_state,
                train_table,
                val_table,
                seed=0,
                phase="outer",
                selection=selection,
            )
        if len(frames) < 2:
            raise ValueError(
                f"Expected a train frame and validation frames; captured {len(frames)}"
            )
        # Some tasks (e.g. F1) have many official validation timestamps. Keep
        # all of them, with their own exact row-level cutoff and original order.
        cutoffs = [cutoffs[0], cutoffs[1:]]
        frames = [frames[0], pd.concat(frames[1:], ignore_index=True)]
        original_score = float(result.full_validation_score)
        cache = {
            "checkpoint_sha256": digest,
            "cutoffs": cutoffs,
            "fold": args.fold,
            "control_score": original_score,
            "plan_fingerprint": result.plan_fingerprint,
        }
        for name, frame in zip(("train", "validation"), frames):
            frame.to_parquet(args.output / f"{name}-frame.parquet", index=False)
        cache_file.write_text(json.dumps(cache, indent=2) + "\n")
        del result
    print(
        json.dumps(
            {
                "phase": "frozen",
                "cutoffs": cutoffs,
                "rows": [len(f) for f in frames],
                "control_score": original_score,
            }
        ),
        flush=True,
    )
    tables = {
        name: EventTable(
            table.df, table.pkey_col, table.time_col, table.fkey_col_to_pkey_table
        )
        for name, table in inner.db_state.table_dict.items()
    }
    supplements, plans = {}, {}
    for mode in atoms:
        # Temporal tests must not gain new depth-two fact paths on a depth-one
        # baseline. Numeric/conditional enrichment deliberately tests bounded
        # extra FK paths; its coverage is recorded separately below.
        max_depth = min(2, trial.config.depth) if mode in {"subday", "clock"} else 2
        plan = EventSummaryPlan.fit(
            tables,
            source.task.entity_table,
            cutoffs[0],
            max_numeric=args.max_numeric,
            max_paths=args.max_paths,
            max_depth=max_depth,
        )
        plans[mode] = {
            **plan.audit(),
            "event_max_depth": max_depth,
            "numeric_scope": args.numeric_scope,
            "numeric_history_days": args.numeric_history_days,
            "max_numeric_pairs": args.max_numeric_pairs,
            "numeric_pair_selection": "balanced_column_coverage",
        }
        supplements[mode] = []
        for frame, cutoff in zip(frames, cutoffs):
            if "__event_cutoff" not in frame:
                if isinstance(cutoff, list) and len(cutoff) != 1:
                    raise ValueError(
                        "Cached multi-cutoff frame lacks row availability timestamps"
                    )
                frame["__event_cutoff"] = (
                    cutoff[0] if isinstance(cutoff, list) else cutoff
                )
            blocks = []
            for stamp, group in frame.groupby("__event_cutoff", sort=False):
                group_block = plan.transform(
                    tables,
                    group["__event_entity"],
                    stamp,
                    mode=mode,
                    numeric_scope=args.numeric_scope,
                    numeric_history_days=args.numeric_history_days,
                    max_numeric_pairs=args.max_numeric_pairs,
                )
                group_block.index = group.index
                blocks.append(group_block)
            block = pd.concat(blocks).sort_index()
            block.columns = [f"supplement_{mode}_{c}" for c in block]
            supplements[mode].append(block.reset_index(drop=True))
        print(
            json.dumps(
                {
                    "phase": "supplement",
                    "mode": mode,
                    "columns": len(supplements[mode][0].columns),
                }
            ),
            flush=True,
        )
    records, predictions = [], {}
    truth = pd.to_numeric(frames[1][target]).to_numpy(dtype=float)
    for arm in ("control", *modes):
        train, validation = [frame.loc[:, columns].copy() for frame in frames]
        if arm != "control":
            for atom in arm.split("+"):
                train = pd.concat([train, supplements[atom][0]], axis=1)
                validation = pd.concat([validation, supplements[atom][1]], axis=1)
        feature_columns = list(train.columns)
        x, vx = [
            prepare_prediction_features(
                frame,
                columns=feature_columns,
                categorical_columns=categorical,
                datetime_columns=datetime,
            )
            for frame in (train, validation)
        ]
        cat_indices = [i for i, column in enumerate(x) if column in categorical]
        seed_predictions = []
        for seed in seeds:
            tx, ty = sample_training_rows(
                x, frames[0][target], limit=100_000, task=task, random_state=seed + 1
            )
            model, seconds, classes = fit_final_catboost(
                tx,
                ty,
                task=task,
                categorical=cat_indices,
                random_state=seed,
                model_params=None,
            )
            metric, score, prediction = score_estimator(
                model, vx, frames[1][target], task=task, target_classes=classes
            )
            if arm == "control" and seed == 0 and abs(score - original_score) > 1e-9:
                raise RuntimeError(
                    f"Matched control does not reproduce the frozen system: {score} vs {original_score}"
                )
            seed_predictions.append(prediction)
            record = {
                "arm": arm,
                "seed": seed,
                "metric": metric,
                "score": score,
                "features": len(feature_columns),
                "model_seconds": seconds,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            np.savez_compressed(
                args.output / f"{arm}-seed{seed}.npz",
                truth=truth,
                prediction=prediction,
            )
        predictions[arm] = np.mean(seed_predictions, axis=0)
    comparison = {}
    control = _prediction_score(truth, predictions["control"], task)
    for arm, prediction in predictions.items():
        score = _prediction_score(truth, prediction, task)
        comparison[arm] = {
            "score": score,
            "improvement": score - control
            if task == "classification"
            else control - score,
        }
        if arm != "control":
            comparison[arm]["improvement_95pct_paired_row_bootstrap"] = (
                _paired_interval(truth, predictions["control"], prediction, task)
            )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "task": f"{args.dataset}/{args.task}",
        "fold": args.fold,
        "scope": "inner validation only, frozen selected graph, no test",
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "frozen_frame_cache": str(cache_root),
        "cutoffs": cutoffs,
        "frame_rows": [len(frame) for frame in frames],
        "budget": {
            "training_frames": 1,
            "estimator_rows": 100_000,
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 10.0,
            "seeds": seeds,
        },
        "event_plans_by_mode": plans,
        "records": records,
        "seed_averaged_prediction_comparison": comparison,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_gib": rss / (1024**3 if sys.platform == "darwin" else 1024**2),
        "versions": {
            name: version(name)
            for name in ("pandas", "numpy", "duckdb", "catboost", "relbench")
        },
        "implementation_sha256": IMPLEMENTATION_SHA256,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
