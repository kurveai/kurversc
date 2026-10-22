"""Freeze DB AND observed-label providers at each retrospective phase start.

Required for multi-cutoff holdouts: their features must NOT acquire later
events or earlier heldout targets just because row timestamps advance.
Only inner-split state is used. Estimators run with one tree to drive ordinary
materialization; their scores are discarded, never used as policy evidence.
"""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import resource
import time

import numpy as np
import pandas as pd

from kurversc.staged_adapter import resolve_cache_config
from kurversc.staged_adapter import (
    _copy_timeless_tables,
    _drop_dangling_seeds,
)
from kurversc.task_source import load_task_source as RelBenchDatasetTask
from kurversc.staged_adapter import KURVERSC_DEFAULTS, _SelectedGraph
from kurversc.plan_dates import atomic_execution_plan
from scripts.benchmark_frozen_event_summaries import _capture
from scripts.run_kurversc_selected_relarena import AuditWinnerSystem
from scripts.temporal_campaign_support import retrospective_folds


def alignment_headers(cache, metadata):
    """Load only fields used to align cohort identities and observed targets.

    Payloads are rematerialized below from the frozen graph. Loading all of
    their historical copies here wastes memory without contributing a value
    to a snapshot, estimator or selection decision.
    """
    columns = ["__event_entity", "__event_cutoff", metadata["target"]]
    return pd.concat([pd.read_parquet(cache / name, columns=columns)
        for name in ("history.parquet", "validation.parquet")], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument("--duckdb-memory-limit", default="6GB")
    parser.add_argument("--duckdb-temp-limit", default="80GB")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.cache / "materialization.json").read_text())
    with Path(metadata["checkpoint"]).open("rb") as handle:
        trial = pickle.load(handle)["trial"]
    trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    original = alignment_headers(args.cache, metadata)
    times = pd.to_datetime(original["__event_cutoff"])
    cuts = sorted(times.unique())
    block_size = metadata["validation_cutoffs"]
    source = RelBenchDatasetTask(metadata["dataset"], metadata["task"])
    inner, task = source.inner_split(), source.task
    labels = pd.concat([inner.train_table.df, inner.eval_table.df], ignore_index=True)
    label_dates = pd.to_datetime(labels[task.time_col])
    boundary_lag = pd.Timestamp(
        pd.to_datetime(inner.eval_table.df[task.time_col]).min()
    ) - pd.Timestamp(inner.cutoff)
    if boundary_lag < pd.Timedelta(0):
        raise ValueError("Prediction cohort starts before the supplied phase boundary")
    KURVERSC_DEFAULTS.update(
        full_training_frames=1,
        full_validation_frames=None,
        base_distribution_features=True,
        temporal_estimator_mode="frame_ensemble",
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_max_temp_directory_size=args.duckdb_temp_limit,
        model_params={"iterations": 1, "thread_count": 1},
    )
    manifests = []
    folds = retrospective_folds(cuts, size=block_size, origins=args.origins)
    if not folds:
        raise ValueError("Insufficient history for two retrospective phases")
    for fold_number, heldout_cuts in enumerate(folds):
        origin = pd.Timestamp(heldout_cuts[0])
        knowledge_cutoff = origin - boundary_lag
        if knowledge_cutoff > pd.Timestamp(inner.cutoff):
            raise ValueError("Retrospective origin exceeds supplied inner DB boundary")
        name = f"fold{fold_number}"
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        previous_path = folder / "manifest.json"
        data_path = folder / "validation.parquet"
        if previous_path.exists() and data_path.exists():
            previous = json.loads(previous_path.read_text())
            if (
                previous["origin"] != str(origin)
                or previous["snapshot_cutoff"] != str(knowledge_cutoff)
                or previous["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                or previous["cutoffs"] != list(map(str, heldout_cuts))
                or previous["feature_cache_sha256"]
                != hashlib.sha256(data_path.read_bytes()).hexdigest()
            ):
                raise ValueError(
                    "Existing phase snapshot has different immutable inputs"
                )
            manifests.append(previous)
            print(json.dumps(dict(phase="snapshot_reused", fold=name)), flush=True)
            continue
        db = inner.db_state.upto(knowledge_cutoff)
        _copy_timeless_tables(db)
        source._dataset.validate_and_correct_db(db)
        train_table, val_table = (
            copy.copy(inner.train_table),
            copy.copy(inner.eval_table),
        )
        train_table.df = labels.loc[
            (label_dates < knowledge_cutoff)
            & (label_dates + task.timedelta <= knowledge_cutoff)
        ].copy()
        if hasattr(source,"mature_labels"):
            train_table.df = source.mature_labels(train_table.df,knowledge_cutoff)
        val_table.df = labels.loc[label_dates.isin(heldout_cuts)].copy()
        count = len(db.table_dict[task.entity_table].df)
        if hasattr(source,"mature_labels"):
            visible_ids = db.table_dict[task.entity_table].df[db.table_dict[task.entity_table].pkey_col]
            for table in (train_table,val_table):
                table.df = table.df.loc[table.df[task.entity_col].isin(visible_ids)].copy()
        else:
            train_table = _drop_dangling_seeds(train_table, task.entity_col, count)
            val_table = _drop_dangling_seeds(val_table, task.entity_col, count)
        system = AuditWinnerSystem(
            cache=resolve_cache_config(None, on_miss="raise"),
            run_identity=source.run_identity(),
        )
        system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
        with _capture(
            metadata["feature_schema"]["columns"],
            task.target_col,
            task.entity_col,
            task.entity_table,
        ) as (frames, frame_cuts):
            fitted, _ = system._fit_arm(
                task,
                db,
                train_table,
                val_table,
                seed=0,
                phase="outer",
                selection=_SelectedGraph(
                    config=trial.config,
                    execution_plan=copy.deepcopy(trial.execution_plan),
                ),
            )
        validation = pd.concat(
            [f for f, c in zip(frames, frame_cuts) if pd.Timestamp(c) in heldout_cuts],
            ignore_index=True,
        )
        key_columns = ["__event_entity", "__event_cutoff"]
        old = original[key_columns].assign(__position=np.arange(len(original)))
        aligned = validation.merge(
            old, on=key_columns, how="left", validate="one_to_one"
        )
        if aligned["__position"].isna().any():
            raise ValueError("Frozen snapshot created unknown heldout rows")
        aligned = aligned.sort_values("__position")
        np.testing.assert_array_equal(
            aligned[task.target_col], original.iloc[aligned.__position][task.target_col]
        )
        aligned.to_parquet(folder / "validation.parquet", index=False)
        manifest = dict(
            origin=str(origin),
            cutoffs=list(map(str, heldout_cuts)),
            checkpoint_sha256=metadata["checkpoint_sha256"],
            inner_db_cutoff=str(inner.cutoff),
            snapshot_cutoff=str(knowledge_cutoff),
            phase_boundary_lag=str(boundary_lag),
            observed_training_label_rows=len(train_table.df),
            latest_observed_label_complete_at=str(
                pd.to_datetime(train_table.df[task.label_available_at]).max()
                if getattr(task,"label_available_at",None)
                else (pd.to_datetime(train_table.df[task.time_col]) + task.timedelta).max()
            ),
            database_rows={
                name: len(table.df) for name, table in db.table_dict.items()
            },
            heldout_rows=len(aligned),
            feature_cache_sha256=hashlib.sha256(
                (folder / "validation.parquet").read_bytes()
            ).hexdigest(),
            scope="One-tree materialization only; estimator scores discarded. DB and target/peer providers frozen at retrospective phase start.",
        )
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(manifest)
        print(
            json.dumps(dict(phase="snapshot_materialized", fold=name, **manifest)),
            flush=True,
        )
        del frames, fitted, db, validation, aligned
        gc.collect()
    (args.output / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    (args.output / "resources.json").write_text(
        json.dumps(
            dict(
                elapsed_seconds=time.perf_counter() - started,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**3,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
